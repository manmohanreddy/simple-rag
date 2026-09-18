import logging

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException

from . import config
from .query import answer_with_meta
from .store import get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_rag.api")

app = FastAPI(title="simple-rag API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    timings: dict
    warning: str | None = None


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not config.API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY is not set")
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@app.get("/health")
def health():
    try:
        get_client().get_collections()
        return {"status": "ok", "qdrant": "reachable"}
    except Exception as e:
        logger.warning("health check: qdrant unreachable: %s", e)
        return {"status": "degraded", "qdrant": "unreachable"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    try:
        result = answer_with_meta(req.question)
    except ResponseHandlingException as e:
        logger.error("qdrant unreachable: %s", e)
        raise HTTPException(status_code=503, detail="Vector store unreachable") from e
    except anthropic.RateLimitError as e:
        logger.warning("claude rate limited: %s", e)
        raise HTTPException(status_code=429, detail="Upstream rate limited, retry shortly") from e
    except anthropic.APIConnectionError as e:
        logger.error("claude connection/timeout error: %s", e)
        raise HTTPException(status_code=503, detail="Upstream LLM unreachable or timed out") from e
    except anthropic.APIStatusError as e:
        logger.error("claude API error %s: %s", e.status_code, e.message)
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e.message}") from e

    if result["response"] is None:
        raise HTTPException(status_code=503, detail=result["text"])

    if result["stop_reason"] == "refusal":
        logger.warning("claude refused question=%r", req.question)
        raise HTTPException(status_code=502, detail="Model declined to answer this question")

    warning = None
    if result["stop_reason"] == "max_tokens":
        warning = "Response was truncated (hit max_tokens) - answer may be incomplete"
        logger.warning("response truncated question=%r", req.question)

    sources = sorted({h.payload["source"] for h in result["hits"]})

    logger.info(
        "query ok question=%r retrieval_s=%.3f llm_s=%.3f total_s=%.3f",
        req.question,
        result["timings"]["retrieval_total_s"],
        result["timings"]["llm_s"],
        result["timings"]["total_s"],
    )

    return QueryResponse(answer=result["text"], sources=sources, timings=result["timings"], warning=warning)
