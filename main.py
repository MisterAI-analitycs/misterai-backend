"""MisterAI backend (FastAPI).

Orquesta la subida de vídeos a Cloudflare R2 (multipart con URLs firmadas), registra los partidos
en Supabase y avisa al worker de Modal cuando un partido está listo para analizarse.
No procesa vídeo. Toda la configuración llega por variables de entorno (ver .env.example).
"""
import math
import os
import uuid

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


SUPABASE_URL = env("SUPABASE_URL").rstrip("/")
SUPABASE_ANON_KEY = env("SUPABASE_ANON_KEY")
SUPABASE_SECRET_KEY = env("SUPABASE_SECRET_KEY")
R2_BUCKET = env("R2_BUCKET")
ALLOWED_ORIGINS = [o.strip() for o in env("ALLOWED_ORIGINS", "").split(",") if o.strip()]
MAX_FILE_BYTES = int(env("MAX_FILE_GB", "6")) * 1024**3
PART_SIZE = 64 * 1024**2          # 64 MB por trozo (R2 exige >= 5 MB salvo el último)
URL_TTL = int(env("URL_TTL_SECONDS", "3600"))
ALLOWED_EXT = {"mp4", "mov", "m4v"}

# Worker de Modal (opcionales: si faltan, el partido queda 'pendiente' sin avisar)
WORKER_URL = env("WORKER_URL", "").rstrip("/")
WORKER_TOKEN = env("WORKER_TOKEN", "")

s3 = boto3.client(
    "s3",
    endpoint_url=env("R2_ENDPOINT"),   # sin el nombre del bucket al final
    aws_access_key_id=env("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
    region_name="auto",
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

app = FastAPI(title="MisterAI backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------- Supabase ----------
def admin_headers() -> dict:
    """Cabeceras con la clave secreta. Las claves nuevas (sb_secret_...) no son JWT."""
    h = {"apikey": SUPABASE_SECRET_KEY, "Content-Type": "application/json"}
    if SUPABASE_SECRET_KEY.startswith("eyJ"):  # clave legacy service_role (JWT)
        h["Authorization"] = f"Bearer {SUPABASE_SECRET_KEY}"
    return h


def current_user(authorization: str = Header(default="")) -> str:
    """Valida la sesión del entrenador contra Supabase Auth y devuelve su user_id."""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Falta el token de sesión")
    token = authorization[7:]
    r = httpx.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if r.status_code != 200:
        raise HTTPException(401, "Sesión no válida o caducada")
    return r.json()["id"]


def find_duplicate(user_id: str, huella: str) -> dict | None:
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/partidos",
        params={"user_id": f"eq.{user_id}", "huella": f"eq.{huella}", "limit": "1"},
        headers=admin_headers(),
        timeout=10,
    )
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def insert_partido(row: dict) -> dict:
    headers = {**admin_headers(), "Prefer": "return=representation"}
    r = httpx.post(f"{SUPABASE_URL}/rest/v1/partidos", json=row, headers=headers, timeout=10)
    if r.status_code >= 300:
        raise HTTPException(502, "No se pudo registrar el partido")
    return r.json()[0]


def require_own_key(key: str, user_id: str) -> None:
    """Un usuario solo puede tocar objetos de su propia carpeta en R2."""
    if not key.startswith(f"{user_id}/") or ".." in key:
        raise HTTPException(403, "Clave de vídeo no permitida")


def avisar_worker(partido_id: str) -> None:
    """Avisa al worker de Modal. Si falla, el partido se queda en 'pendiente' y se puede relanzar."""
    if not WORKER_URL or not WORKER_TOKEN:
        return
    try:
        httpx.post(
            f"{WORKER_URL}/procesar",
            json={"partido_id": partido_id},
            headers={"x-worker-token": WORKER_TOKEN},
            timeout=60,  # el primer arranque del contenedor de Modal puede tardar
        )
    except httpx.HTTPError:
        pass


# ---------- Modelos ----------
class InitReq(BaseModel):
    filename: str = Field(max_length=300)
    size: int = Field(gt=0)
    content_type: str = Field(default="video/mp4", max_length=100)
    huella: str | None = Field(default=None, max_length=128)


class PartDone(BaseModel):
    PartNumber: int = Field(ge=1, le=10000)
    ETag: str = Field(max_length=200)


class CompleteReq(BaseModel):
    key: str = Field(max_length=300)
    upload_id: str = Field(max_length=1000)
    parts: list[PartDone]
    nombre: str = Field(default="", max_length=120)
    modalidad: str = Field(default="Fútbol 11", max_length=40)
    huella: str | None = Field(default=None, max_length=128)


class AbortReq(BaseModel):
    key: str = Field(max_length=300)
    upload_id: str = Field(max_length=1000)


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/upload/init")
def upload_init(body: InitReq, user_id: str = Depends(current_user)):
    """Inicia la subida multipart y devuelve una URL firmada por cada trozo."""
    ext = body.filename.rsplit(".", 1)[-1].lower() if "." in body.filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Formato no válido (.mp4, .mov o .m4v)")
    if body.size > MAX_FILE_BYTES:
        raise HTTPException(413, f"El vídeo supera el máximo de {MAX_FILE_BYTES // 1024**3} GB")

    if body.huella:
        dup = find_duplicate(user_id, body.huella)
        if dup:
            return {"duplicado": True, "partido": dup}

    key = f"{user_id}/{uuid.uuid4().hex}.{ext}"
    n_parts = math.ceil(body.size / PART_SIZE)
    try:
        mp = s3.create_multipart_upload(Bucket=R2_BUCKET, Key=key, ContentType=body.content_type)
        parts = [
            {
                "part_number": i,
                "url": s3.generate_presigned_url(
                    "upload_part",
                    Params={"Bucket": R2_BUCKET, "Key": key, "UploadId": mp["UploadId"], "PartNumber": i},
                    ExpiresIn=URL_TTL,
                ),
            }
            for i in range(1, n_parts + 1)
        ]
    except ClientError:
        raise HTTPException(502, "No se pudo iniciar la subida")
    return {"duplicado": False, "key": key, "upload_id": mp["UploadId"], "part_size": PART_SIZE, "parts": parts}


@app.post("/api/upload/complete")
def upload_complete(body: CompleteReq, background: BackgroundTasks, user_id: str = Depends(current_user)):
    """Cierra la subida en R2, registra el partido como 'pendiente' y avisa al worker."""
    require_own_key(body.key, user_id)
    parts = sorted(body.parts, key=lambda p: p.PartNumber)
    try:
        s3.complete_multipart_upload(
            Bucket=R2_BUCKET,
            Key=body.key,
            UploadId=body.upload_id,
            MultipartUpload={"Parts": [{"PartNumber": p.PartNumber, "ETag": p.ETag} for p in parts]},
        )
        size = s3.head_object(Bucket=R2_BUCKET, Key=body.key)["ContentLength"]
    except ClientError:
        raise HTTPException(502, "No se pudo completar la subida")

    if size > MAX_FILE_BYTES:  # el cliente mintió sobre el tamaño
        s3.delete_object(Bucket=R2_BUCKET, Key=body.key)
        raise HTTPException(413, "El vídeo supera el tamaño máximo")

    partido = insert_partido({
        "user_id": user_id,
        "nombre": body.nombre.strip() or "Partido sin nombre",
        "modalidad": body.modalidad,
        "file_path": body.key,
        "huella": body.huella,
        "tamano_bytes": size,
        "estado_analisis": "pendiente",
    })
    background.add_task(avisar_worker, partido["id"])  # responde ya; el aviso se hace después
    return {"duplicado": False, "partido": partido}


@app.post("/api/upload/abort")
def upload_abort(body: AbortReq, user_id: str = Depends(current_user)):
    require_own_key(body.key, user_id)
    try:
        s3.abort_multipart_upload(Bucket=R2_BUCKET, Key=body.key, UploadId=body.upload_id)
    except ClientError:
        pass  # si ya no existe, da igual
    return {"ok": True}
