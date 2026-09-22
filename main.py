import time
import logging

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xgboost

# Ensure torch and transformers (and Triton) are loaded BEFORE TensorFlow/Keras
# to prevent native library symbol conflicts on WSL/CUDA.
# Skip entirely on macOS — importing torch loads native OpenMP libs that
# cannot be unloaded and cause a segfault when XGBoost loads its own copy.
import platform
if platform.system() != "Darwin":
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import triton  # type: ignore
            except ImportError:
                pass
            import transformers
    except ImportError:
        pass

from contextlib import asynccontextmanager

from fastapi import FastAPI

# Add agent-orchestration to sys.path so that conversation.*, orchestrator.*,
# agents.*, schemas.*, and location.* are importable as root-level packages.
import sys
from pathlib import Path
_AGENT_DIR = Path(__file__).resolve().parents[1] / "agent-orchestration"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from backend.api.routes.orca import router as orca_router
from backend.api.routes.alerts import router as alerts_router
from backend.core.config import MODEL_BACKEND
from backend.services.orca_service import OrcaService, OrcaSessionStore



logger = logging.getLogger(__name__)

_startup_time: float = 0.0
_total_requests: int = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    logger.info("STARTUP: entering lifespan")

    logger.info("STARTUP: loading Qwen model (auto-detecting backend)...")

    from conversation.model import get_conversation_model

    model = get_conversation_model()

    backend_name = getattr(model, "backend", "cuda")
    logger.info(f"STARTUP: Qwen loaded (backend={backend_name})")
    
    logger.info("STARTUP: Warming up Qwen model caches...")
    try:
        from conversation.prompts import EXTRACTION_SYSTEM_PROMPT, RESPONSE_SYSTEM_PROMPT_TEMPLATE
        model.extract(EXTRACTION_SYSTEM_PROMPT, "warmup query test")
        model.generate_text(RESPONSE_SYSTEM_PROMPT_TEMPLATE.format(language_desc="English."), "warmup")
        logger.info("STARTUP: Qwen warmup complete")
    except Exception as e:
        logger.error(f"STARTUP: Failed to warm up Qwen model: {e}")

    logger.info("STARTUP: creating engine")

    from backend.dependencies.engine import create_orca_engine
    engine = create_orca_engine()

    logger.info("STARTUP: engine created")

    sessions = OrcaSessionStore()

    app.state.orca_model = model
    app.state.orca_engine = engine
    app.state.orca_service = OrcaService(
        model,
        engine,
        sessions,
    )

    logger.info("STARTUP: service ready, yielding to FastAPI")

    yield

    logger.info("SHUTDOWN: cleaning up")

    if MODEL_BACKEND == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            pass


app = FastAPI(
    title="ORCA API",
    description="Marine Ecosystem Reasoning with Collaborative Agents",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings

if settings.ORCA_COMMAND_CENTER_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.ORCA_COMMAND_CENTER_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/api/v1/health")
def health_check():
    import backend.api.dependencies.rate_limit as rl
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0.0
    return {
        "status": "ok",
        "service": "ORCA API",
        "model_backend": MODEL_BACKEND,
        "uptime_seconds": uptime,
        "active_concurrent_requests": rl._active_requests,
    }


app.include_router(orca_router)
app.include_router(alerts_router)