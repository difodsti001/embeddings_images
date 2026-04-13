"""
app.py — API REST para el motor de vectorizacion ColPali

Endpoints:
  POST /api/index              → sube un PDF, lo vectoriza e inserta en Qdrant
  POST /api/search/text        → busqueda por texto (devuelve hits)
  POST /api/search/image       → busqueda por imagen (devuelve hits)
  POST /api/search/answer      → busqueda por texto + respuesta LLM (flujo completo)
  GET  /api/collections        → lista colecciones
  GET  /api/collections/{name} → info de una coleccion
  GET  /health                 → estado del servicio
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import core

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: pre-carga del modelo al arrancar
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servicio ColPali...")
    try:
        core.get_model()
        core.get_qdrant()
        logger.info("Servicio listo.")
    except Exception as exc:
        logger.error("Error al inicializar dependencias: %s", exc)
    yield
    logger.info("Servicio detenido.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ColPali Vector Engine",
    description="Motor de vectorizacion de documentos PDF con ColPali + Qdrant + LLM.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IndexResponse(BaseModel):
    status: str
    collection: str
    filename: str
    pages_indexed: int
    elapsed_seconds: float


class SearchHit(BaseModel):
    id: int | str
    score: float
    image_name: Optional[str] = None
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    total_pages: Optional[int] = None
    image_base64: Optional[str] = Field(None, description="Pagina en base64 PNG")


class SearchResponse(BaseModel):
    query_type: str
    collection: str
    limit: int
    results: list[SearchHit]


class AnswerResponse(BaseModel):
    query: str
    collection: str
    answer: Optional[str] = Field(None, description="Respuesta generada por el LLM")
    provider: Optional[str] = Field(None, description="LLM usado: 'gpt' o 'gemini'")
    error: Optional[str] = None
    results: list[SearchHit] = Field(default_factory=list, description="Paginas de contexto usadas")


class CollectionInfo(BaseModel):
    name: str
    vectors_count: Optional[int] = None
    points_count: Optional[int] = None
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Estado"])
def health():
    return {"status": "ok", "model": core.COLPALI_MODEL_NAME}


# ── Colecciones ──────────────────────────────────────────────────────────────

@app.get("/api/collections", response_model=list[str], tags=["Colecciones"])
def get_collections():
    """Lista todas las colecciones disponibles en Qdrant."""
    try:
        return core.list_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error al conectar con Qdrant: {exc}")


@app.get("/api/collections/{collection_name}", response_model=CollectionInfo, tags=["Colecciones"])
def get_collection_info(collection_name: str):
    """Retorna metadata de una coleccion."""
    try:
        return core.collection_info(collection_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Coleccion no encontrada: {exc}")


# ── Indexacion ───────────────────────────────────────────────────────────────

@app.post("/api/index", response_model=IndexResponse, tags=["Indexacion"])
async def index_pdf(
    file: Annotated[UploadFile, File(description="Archivo PDF a indexar")],
    collection_name: Annotated[str, Form()] = core.DEFAULT_COLLECTION,
):
    """
    Recibe un PDF, lo convierte a imagenes (300 dpi), genera embeddings ColPali
    por cada pagina e inserta los vectores en la coleccion indicada.
    La coleccion debe existir previamente.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="El archivo esta vacio.")
    try:
        result = core.index_pdf(pdf_bytes=pdf_bytes, collection_name=collection_name, filename=file.filename)
        return IndexResponse(status="ok", **result)
    except Exception as exc:
        logger.exception("Error al indexar PDF")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Busqueda por texto ────────────────────────────────────────────────────────

@app.post("/api/search/text", response_model=SearchResponse, tags=["Busqueda"])
async def search_text(
    query: Annotated[str, Form()],
    collection_name: Annotated[str, Form()] = core.DEFAULT_COLLECTION,
    limit: Annotated[int, Form(ge=1, le=50)] = core.SEARCH_LIMIT,
):
    """Busca paginas relevantes usando una consulta de texto."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="El query no puede estar vacio.")
    try:
        hits = core.search_by_text(query, collection_name, limit)
        return SearchResponse(
            query_type="text",
            collection=collection_name,
            limit=limit,
            results=[SearchHit(**h) for h in hits],
        )
    except Exception as exc:
        logger.exception("Error en busqueda por texto")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Busqueda por imagen ───────────────────────────────────────────────────────

@app.post("/api/search/image", response_model=SearchResponse, tags=["Busqueda"])
async def search_image(
    file: Annotated[UploadFile, File(description="Imagen de consulta")],
    collection_name: Annotated[str, Form()] = core.DEFAULT_COLLECTION,
    limit: Annotated[int, Form(ge=1, le=50)] = core.SEARCH_LIMIT,
):
    """Busca paginas similares usando una imagen como consulta."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="El archivo esta vacio.")
    try:
        hits = core.search_by_image_bytes(image_bytes, collection_name, limit)
        return SearchResponse(
            query_type="image",
            collection=collection_name,
            limit=limit,
            results=[SearchHit(**h) for h in hits],
        )
    except Exception as exc:
        logger.exception("Error en busqueda por imagen")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Busqueda + respuesta LLM (flujo completo, solo texto) ────────────────────

@app.post("/api/search/answer", response_model=AnswerResponse, tags=["Busqueda"])
async def search_and_answer(
    query: Annotated[str, Form(description="Pregunta del usuario")],
    collection_name: Annotated[str, Form()] = core.DEFAULT_COLLECTION,
    limit: Annotated[int, Form(ge=1, le=10)] = 2,
):
    """
    Flujo completo de busqueda por texto con respuesta LLM:
      1. Vectoriza la pregunta con ColPali
      2. Recupera las top-N paginas mas relevantes de Qdrant (defecto: 2)
      3. Envia la pregunta + imagenes de las paginas a GPT-4o-mini
      4. Si GPT falla, hace fallback a Gemini 1.5 Flash
      5. Devuelve la respuesta del LLM junto con las paginas de referencia
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="El query no puede estar vacio.")
    try:
        # Paso 1+2: busqueda vectorial
        hits = core.search_by_text(query, collection_name, limit)

        # Paso 3+4: respuesta LLM
        llm_result = core.answer_with_llm(query, hits)

        return AnswerResponse(
            query=query,
            collection=collection_name,
            answer=llm_result.get("answer"),
            provider=llm_result.get("provider"),
            error=llm_result.get("error"),
            results=[SearchHit(**h) for h in hits],
        )
    except Exception as exc:
        logger.exception("Error en search/answer")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
