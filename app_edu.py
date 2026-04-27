"""
app_edu.py — API REST Buscador de Recursos Educativos MINEDU v2

Endpoints:
  GET  /health
  GET  /api/edu/resources                -> lista recursos
  GET  /api/edu/resources/filters        -> opciones de filtros
  POST /api/edu/repository/scan          -> escanear carpeta
  POST /api/edu/repository/index         -> indexar pendientes locales
  POST /api/edu/repository/upload        -> subir PDF manualmente + metadata
  POST /api/edu/repository/excel/stream  -> procesar Excel MINEDU con progreso SSE
  POST /api/edu/search/text              -> buscar por texto
  POST /api/edu/search/image             -> buscar por imagen
  POST /api/edu/search/similar           -> buscar paginas similares
"""

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated, Optional
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

import core
import core_edu

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando Buscador de Recursos Educativos...")
    try:
        core.get_model()
        core.get_qdrant()
    except Exception as exc:
        logger.error("Error al inicializar: %s", exc)
    yield


app = FastAPI(title="Buscador Recursos Educativos MINEDU", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchHit(BaseModel):
    id:           int | str
    score:        float
    image_base64: Optional[str] = None
    image_name:   Optional[str] = None
    source_file:  Optional[str] = None
    page_number:  Optional[int] = None
    total_pages:  Optional[int] = None
    # Filtros
    tipo_recurso: Optional[str] = None
    nivel:        Optional[str] = None
    area:         Optional[str] = None
    categoria:    Optional[str] = None
    # Detalle
    titulo:             Optional[str] = None
    sub_categoria:      Optional[str] = None
    modalidad:          Optional[str] = None
    servicio_educativo: Optional[str] = None
    autor:              Optional[str] = None
    derecho_autoridad:  Optional[str] = None
    anio_edicion:       Optional[str] = None
    lengua_idioma:      Optional[str] = None
    resumen:            Optional[str] = None
    competencias:       Optional[str] = None


class SearchResponse(BaseModel):
    query_type:      str
    limit:           int
    results:         list[SearchHit]
    filters_applied: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Estado"])
def health():
    return {"status": "ok", "model": core.COLPALI_MODEL_NAME, "collection": core_edu.EDU_COLLECTION}


@app.post("/api/edu/repository/upload", tags=["Repositorio"])
async def upload_manual(
    file:         Annotated[UploadFile, File()],
    titulo:       Annotated[str,  Form()] = "",
    tipo_recurso: Annotated[str,  Form()] = "",
    nivel:        Annotated[str,  Form()] = "",
    area:         Annotated[str,  Form()] = "",
    categoria:    Annotated[str,  Form()] = "",
    autor:        Annotated[str,  Form()] = "",
    anio_edicion: Annotated[str,  Form()] = "",
    resumen:      Annotated[str,  Form()] = "",
):
    """Sube un PDF con metadata ingresada manualmente."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Archivo vacio.")
    metadata = {
        "titulo": titulo, "tipo_recurso": tipo_recurso, "nivel": nivel,
        "area": area, "categoria": categoria, "autor": autor,
        "anio_edicion": anio_edicion, "resumen": resumen,
    }
    try:
        result = core_edu.index_manual_upload(pdf_bytes, file.filename, metadata)
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/edu/repository/excel/stream", tags=["Repositorio"])
async def process_excel_stream(
    file: Annotated[UploadFile, File(description="Excel MINEDU (.xlsx)")],
    max_concurrent: Annotated[int, Form(ge=1, le=8)] = 4,
):
    """
    Procesa el Excel MINEDU con progreso en tiempo real via Server-Sent Events.
    Filtra: LENGUA/IDIOMA contiene 'Castellano', TIPO ENLACE == 'PDF', ESTADO == 'Activo'.
    Descarga PDFs en paralelo y los vectoriza con toda la metadata del Excel.
    """
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Solo se aceptan archivos Excel (.xlsx).")

    excel_bytes = await file.read()
    if not excel_bytes:
        raise HTTPException(400, "Archivo vacio.")

    # Guardar Excel temporalmente
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.write(excel_bytes)
    tmp.close()
    tmp_path = tmp.name

    async def event_stream():
        try:
            async for event in core_edu.process_excel_minedu(tmp_path, max_concurrent):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/edu/repository/cancel", tags=["Repositorio"])
async def cancel_excel_processing():
    """
    Cancela el procesamiento en curso del Excel MINEDU.
    Detiene la descarga y vectorización de recursos pendientes.
    """
    try:
        core_edu.cancel_processing()
        return {"status": "ok", "message": "Procesamiento cancelado"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Busqueda ──────────────────────────────────────────────────────────────────

def _make_search_response(query_type, limit, hits, filters):
    return SearchResponse(
        query_type=query_type, limit=limit,
        results=[SearchHit(**h) for h in hits],
        filters_applied={k: v for k, v in filters.items() if v},
    )


@app.post("/api/edu/search/text", response_model=SearchResponse, tags=["Busqueda"])
async def search_text(
    query:        Annotated[str, Form()],
    limit:        Annotated[int, Form(ge=1, le=20)] = core_edu.EDU_SEARCH_LIMIT,
    tipo_recurso: Annotated[Optional[str], Form()] = None,
    nivel:        Annotated[Optional[str], Form()] = None,
    area:         Annotated[Optional[str], Form()] = None,
    categoria:    Annotated[Optional[str], Form()] = None,
):
    if not query.strip():
        raise HTTPException(400, "Query vacio.")
    try:
        hits = core_edu.search_by_text(query, limit, tipo_recurso, nivel, area, categoria)
        return _make_search_response("text", limit, hits,
                                     {"tipo_recurso": tipo_recurso, "nivel": nivel, "area": area, "categoria": categoria})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/edu/search/image", response_model=SearchResponse, tags=["Busqueda"])
async def search_image(
    file:         Annotated[UploadFile, File()],
    limit:        Annotated[int, Form(ge=1, le=20)] = core_edu.EDU_SEARCH_LIMIT,
    tipo_recurso: Annotated[Optional[str], Form()] = None,
    nivel:        Annotated[Optional[str], Form()] = None,
    area:         Annotated[Optional[str], Form()] = None,
    categoria:    Annotated[Optional[str], Form()] = None,
):
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(400, "Archivo vacio.")
    try:
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        hits  = core_edu.search_by_image(image, limit, tipo_recurso, nivel, area, categoria)
        return _make_search_response("image", limit, hits,
                                     {"tipo_recurso": tipo_recurso, "nivel": nivel, "area": area, "categoria": categoria})
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/edu/search/similar", response_model=SearchResponse, tags=["Busqueda"])
async def search_similar(
    image_base64: Annotated[str, Form()],
    limit:        Annotated[int, Form(ge=1, le=20)] = core_edu.EDU_SEARCH_LIMIT,
):
    """Busca paginas visualmente similares dado el base64 PNG de una pagina."""
    if not image_base64.strip():
        raise HTTPException(400, "image_base64 vacio.")
    try:
        hits = core_edu.search_similar_to_page(image_base64.strip(), limit)
        return _make_search_response("similar", limit, hits, {})
    except Exception as exc:
        logger.exception("Error en search/similar")
        raise HTTPException(500, str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_edu:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
