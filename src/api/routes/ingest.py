import os
import shutil
import aiofiles
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from src.ingestion.pipeline import IngestionPipeline
from src.api.deps import get_ingestion_pipeline
from src.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

TEMP_UPLOAD_DIR = "data/uploads"

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> dict[str, Any]:
    """Uploads a document and processes it through the ingestion pipeline."""
    # Ensure temporary upload directory exists
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    
    file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename or "uploaded_doc")

    try:
        # Asynchronously save uploaded file to disk
        async with aiofiles.open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                await buffer.write(chunk)

        logger.info("received_file_for_ingestion", filename=file.filename, saved_path=file_path)

        # Run ingestion pipeline
        stats = await pipeline.ingest_files([file_path])
        return stats

    except Exception as e:
        logger.error("file_upload_ingest_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        # Clean up temporary saved file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as ex:
                logger.warning("failed_to_cleanup_temp_file", path=file_path, error=str(ex))
from typing import Any
