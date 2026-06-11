import json
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.generation.chain import RAGChain
from src.api.deps import get_rag_chain
from src.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    question: str = Field(..., description="The query question to ask the documents.")
    top_k: int = Field(5, description="Number of context chunks to retrieve.")
    alpha: float = Field(0.5, description="Hybrid search alpha (0=BM25 only, 1=vector only).")
    model: str = Field("default", description="Model identifier: gpt-4o-mini | claude-3-haiku-20240307 | ollama/llama3.2")
    stream: bool = Field(False, description="Whether to stream the response back using Server-Sent Events.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata key-value filters to restrict document context.")

@router.post("/query")
async def query_documents(
    request: QueryRequest,
    chain: RAGChain = Depends(get_rag_chain)
) -> Any:
    """Query the RAG system to generate an answer with grounded citations."""
    try:
        if request.stream:
            async def event_generator():
                try:
                    async for chunk in chain.stream_query(
                        question=request.question,
                        top_k=request.top_k,
                        alpha=request.alpha,
                        model=request.model,
                        filters=request.filters
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as ex:
                    logger.error("stream_error", error=str(ex))
                    yield f"data: {json.dumps({'error': str(ex)})}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            response = await chain.query(
                question=request.question,
                top_k=request.top_k,
                alpha=request.alpha,
                model=request.model,
                filters=request.filters
            )
            return response

    except Exception as e:
        logger.error("query_route_failed", question=request.question, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
