"""
backend/main.py
MUSIA - Multilingual Story Illustration
Phase 2+: FastAPI Backend Server — Multilingual Processing (English, Hindi, Bengali)
           with Quantum-Enhanced PQC Inference Pipeline
"""

import os
import re
import html
import time
import asyncio
from collections import defaultdict

from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .story_parser import (
    parse_story_to_scenes,
    parse_story_scenes_detailed,
    detect_language,
    normalize_multilingual_text,
)
from .generate_on_colab import (
    generate_scene_image,
    generate_mock_scene_image,
    get_colab_url,
    set_colab_url,
    ping_colab_tunnel,
)
from .quantum_pipeline import weight_manager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Security & Sanitization Utilities
# ---------------------------------------------------------------------------

MAX_STORY_LENGTH = 5000
MAX_SCENES = 12

def sanitize_text(text: str) -> str:
    """
    Sanitize text input by removing dangerous script/style/iframe tags
    and normalizing unescaped HTML characters.
    """
    if not text:
        return ""
    # Strip script, style, iframe, object tags and their contents
    cleaned = re.sub(r'<(script|style|iframe|object|embed)[^>]*>.*?</\1>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Strip remaining HTML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Unescape HTML entities
    cleaned = html.unescape(cleaned)
    # Strip non-printable ASCII control characters except \n, \r, \t
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MUSIA API",
    description=(
        "Multilingual Story Illustration — Quantum-Enhanced Neural Diffusion.\n\n"
        "Supports English, Hindi (Devanagari), and Bengali (Bangla) narratives. "
        "Uses trained 4-qubit PQC Enhancer (`best_pqc_enhancer_fulldataset.pt`) "
        "and SDXL LoRA adapter (`adapter_model.safetensors`) for inference."
    ),
    version="3.0.0",
)

# Per-IP sliding window rate limiting (max 120 req / min per IP)
RATE_LIMIT_MAX_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60
_request_records = defaultdict(list)

@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    # Determine client IP address
    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    records = _request_records[client_ip]
    _request_records[client_ip] = [ts for ts in records if ts > window_start]

    if request.url.path.startswith("/api/"):
        if len(_request_records[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down and try again."},
                headers={"Retry-After": "60"}
            )
        _request_records[client_ip].append(now)

    response = await call_next(request)

    # Security Headers (CSP, X-Content-Type-Options, Referrer-Policy, X-Frame-Options)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' https: data: blob:; "
        "img-src 'self' https: data: blob:; "
        "connect-src 'self' https: http:;"
    )
    return response

# Allow all origins (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist and mount them reliably
scenes_static_dir = os.path.join(BASE_DIR, "backend", "static", "generated_scenes")
try:
    os.makedirs(scenes_static_dir, exist_ok=True)
except OSError:
    pass

if os.path.exists(scenes_static_dir):
    app.mount(
        "/static/generated_scenes",
        StaticFiles(directory=scenes_static_dir),
        name="generated_scenes",
    )

# Mount frontend directory for static assets
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount(
        "/frontend",
        StaticFiles(directory=frontend_dir),
        name="frontend",
    )


# ---------------------------------------------------------------------------
# Global generation state
# ---------------------------------------------------------------------------

generation_status: dict = {
    "is_running": False,
    "current_scene": 0,
    "total_scenes": 0,
    "scenes": [],
    "image_urls": [],
    "selected_model": "quantum",  # 'quantum' | 'standard' | 'mock'
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StoryRequest(BaseModel):
    story: str = Field(..., max_length=MAX_STORY_LENGTH, description="Raw story text, capped at 5000 characters.")

    @field_validator("story")
    @classmethod
    def story_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("The 'story' field must not be empty.")
        clean = sanitize_text(v)
        if not clean:
            raise ValueError("The 'story' field contains no valid content after sanitization.")
        if len(clean) > MAX_STORY_LENGTH:
            raise ValueError(f"Story exceeds maximum allowed length of {MAX_STORY_LENGTH:,} characters.")
        return clean


class ParseStoryResponse(BaseModel):
    total_scenes: int
    scenes: list[str]


class GenerationRequest(BaseModel):
    """Optional body for /api/start-generation."""
    model: str = "quantum"  # 'quantum' | 'standard' | 'mock'

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("quantum", "standard", "mock"):
            raise ValueError("model must be 'quantum', 'standard', or 'mock'.")
        return v


class SingleSceneRequest(BaseModel):
    """Body for /api/generate-scene and /api/generate proxy."""
    scene_prompt: str = Field(..., min_length=1, max_length=1500)
    scene_id: int = Field(1, ge=1, le=MAX_SCENES)
    model: str = "quantum"
    seed: Optional[int] = None

    @field_validator("scene_prompt")
    @classmethod
    def clean_scene_prompt(cls, v: str) -> str:
        clean = sanitize_text(v)
        if not clean:
            raise ValueError("Scene prompt cannot be empty.")
        return clean


class ColabConfigRequest(BaseModel):
    """Body for /api/config."""
    colab_url: str = Field(..., max_length=500)


class CompareSceneRequest(BaseModel):
    """Body for /api/compare-scene."""
    scene_prompt: str = Field(..., min_length=1, max_length=1500)
    scene_id: int = Field(1, ge=1, le=MAX_SCENES)
    seed: Optional[int] = None

    @field_validator("scene_prompt")
    @classmethod
    def clean_compare_prompt(cls, v: str) -> str:
        clean = sanitize_text(v)
        if not clean:
            raise ValueError("Scene prompt cannot be empty.")
        return clean


# ---------------------------------------------------------------------------
# Background generation pipeline
# ---------------------------------------------------------------------------

async def run_generation_pipeline(model_choice: str = "quantum") -> None:
    """
    Async background coroutine that iterates through all parsed scenes,
    calls the Google Colab GPU bridge for each one (via asyncio.to_thread so the
    event loop stays unblocked), and accumulates image URLs in
    `generation_status`.

    Args:
        model_choice: 'quantum' to use QuantumFeatureEnhancer;
                      'standard' to use plain SDXL embeddings;
                      'mock' for instant local placeholder images.
    """
    global generation_status

    is_mock     = (model_choice == "mock")
    use_quantum = (model_choice == "quantum")
    scenes = generation_status["scenes"]
    total  = generation_status["total_scenes"]
    print(f"[START] Pipeline starting | model={model_choice} | scenes={total}")

    for idx, scene_prompt in enumerate(scenes, start=1):
        # Allow an external stop signal
        if not generation_status["is_running"]:
            print(f"[STOP] Generation pipeline stopped early at scene {idx}/{total}.")
            break

        generation_status["current_scene"] = idx
        print(f"[SCENE] Processing scene {idx}/{total} ...")

        try:
            if is_mock:
                image_url = await asyncio.to_thread(
                    generate_mock_scene_image, scene_prompt, idx
                )
            else:
                image_url = await asyncio.to_thread(
                    generate_scene_image, scene_prompt, idx, use_quantum
                )
            generation_status["image_urls"].append(image_url)
            print(f"[OK] Scene {idx} complete -> {image_url}")
        except Exception as exc:
            # Fallback on Colab/tunnel failure: produce a mock placeholder
            print(f"[WARN] Scene {idx} Colab/generation error - generating mock fallback: {exc}")
            try:
                fallback_url = await asyncio.to_thread(
                    generate_mock_scene_image, scene_prompt, idx
                )
                generation_status["image_urls"].append(fallback_url)
                print(f"[IMG] Scene {idx} mock fallback saved -> {fallback_url}")
            except Exception as fb_exc:
                error_url = f"/static/generated_scenes/error_scene_{idx}.png"
                generation_status["image_urls"].append(error_url)
                print(f"[ERR] Scene {idx} failed (even mock fallback): {fb_exc}")

    generation_status["is_running"] = False
    print("[DONE] Generation pipeline finished.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Serve MUSIA Frontend Dashboard")
def serve_frontend():
    """Serve the frontend index.html dashboard at the root URL."""
    frontend_index = os.path.join(BASE_DIR, "frontend", "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"status": "success", "message": "MUSIA API v3 is up and running."}


@app.get("/api/health", summary="Health check proxy endpoint")
async def health_check():
    """Health check confirming API status, model weight availability, and Colab GPU state."""
    data = await get_system_status()
    data["status"] = "success"
    return data


@app.post(
    "/api/parse-story",
    response_model=ParseStoryResponse,
    summary="Parse story into cinematic scenes",
)
async def parse_story(request: StoryRequest):
    """
    Accept raw story text, split it into cinematic scene prompts, and
    reset the global generation state so a fresh run can be started.
    """
    global generation_status

    scenes = parse_story_to_scenes(request.story)
    if scenes:
        scenes = scenes[:MAX_SCENES]

    if not scenes:
        raise HTTPException(
            status_code=422,
            detail="Story could not be parsed into any scenes. Please provide more content.",
        )

    # Reset global state for a new story
    generation_status.clear()
    generation_status.update({
        "is_running": False,
        "current_scene": 0,
        "total_scenes": len(scenes),
        "scenes": scenes,
        "image_urls": [],
        "selected_model": "quantum",  # reset to default; overridden at start-generation ('quantum' | 'standard' | 'mock')
    })

    return ParseStoryResponse(total_scenes=len(scenes), scenes=scenes)


@app.post(
    "/api/parse-story-detailed",
    summary="Parse story with language detection metadata",
)
async def parse_story_detailed(request: StoryRequest):
    """
    Accept raw story text in English, Hindi, or Bengali and return detailed
    scene metadata including language detection, token counts, and scene titles.
    Supports Purna Viram '।' sentence boundaries for Indic scripts.
    Capped at MAX_SCENES (12 scenes).
    """
    details = parse_story_scenes_detailed(request.story)
    if details:
        details = details[:MAX_SCENES]
    if not details:
        raise HTTPException(
            status_code=422,
            detail="Story could not be parsed into any scenes. Please provide more content.",
        )
    detected_lang = detect_language(normalize_multilingual_text(request.story))
    lang_names = {"en": "English", "hi": "Hindi (Devanagari)", "bn": "Bengali (Bangla)", "mixed": "Multilingual"}
    return {
        "total_scenes": len(details),
        "detected_language": detected_lang,
        "language_name": lang_names.get(detected_lang, "English"),
        "scenes": details,
    }


@app.get("/api/model-info", summary="Trained model weight info")
async def model_info():
    """
    Returns status of the loaded trained model weights:
    - Quantum PQC Enhancer (`best_pqc_enhancer_fulldataset.pt`)
    - SDXL LoRA Adapter (`adapter_model.safetensors`)
    """
    return {
        "quantum_pqc_enhancer": {
            "loaded": weight_manager.is_pqc_loaded,
            "architecture": "4-qubit PQC (AngleEmbedding + BasicEntanglerLayers)",
            "layers": {"n_qubits": 4, "n_layers": 2, "embed_dim": 2048},
            "file": "best_pqc_enhancer_fulldataset.pt",
        },
        "lora_adapter": {
            "available": weight_manager.has_lora_adapter,
            "type": "SDXL UNet LoRA (rank 32)",
            "total_tensors": 1120,
            "file": "adapter_model.safetensors",
        },
        "device": weight_manager.device,
        "inference_modes": ["quantum", "standard", "mock"],
        "supported_languages": [
            {"code": "en", "name": "English", "script": "Latin"},
            {"code": "hi", "name": "Hindi", "script": "Devanagari (U+0900-U+097F)"},
            {"code": "bn", "name": "Bengali", "script": "Bangla (U+0980-U+09FF)"},
        ],
    }


@app.post("/api/start-generation", summary="Start Colab image generation")
async def start_generation(
    background_tasks: BackgroundTasks,
    request: GenerationRequest = GenerationRequest(),
):
    """
    Validate that scenes have been parsed and no pipeline is running,
    then kick off the background generation coroutine.

    Accepts optional JSON body: ``{"model": "quantum" | "standard"}``
    """
    global generation_status

    if not generation_status["scenes"]:
        raise HTTPException(
            status_code=400,
            detail="No scenes found. Please call /api/parse-story first.",
        )

    if generation_status["is_running"]:
        raise HTTPException(
            status_code=409,
            detail="Generation is already in progress.",
        )

    # Store model choice and reset image results
    model_choice = request.model
    generation_status["selected_model"] = model_choice
    generation_status["is_running"]    = True
    generation_status["current_scene"] = 0
    generation_status["image_urls"]    = []

    background_tasks.add_task(run_generation_pipeline, model_choice)

    model_label = {
        "quantum":  "Quantum-Enhanced SDXL",
        "standard": "Standard SDXL",
        "mock":     "Fast Local Preview (Mock)",
    }.get(model_choice, model_choice)

    return {
        "status": "started",
        "model": model_choice,
        "message": (
            f"Generation pipeline started for {generation_status['total_scenes']} scene(s) "
            f"using {model_label}."
        ),
    }


@app.get("/api/generation-status", summary="Poll generation progress")
async def get_generation_status():
    """Return the current generation state (polling endpoint for the frontend)."""
    return generation_status


@app.post("/api/stop-generation", summary="Stop generation pipeline")
async def stop_generation():
    """
    Signal the running pipeline to stop after the current scene finishes.
    The pipeline checks `is_running` before each scene iteration.
    """
    global generation_status

    if not generation_status["is_running"]:
        return {"status": "idle", "message": "No generation pipeline is currently running."}

    generation_status["is_running"] = False
    return {
        "status": "stopping",
        "message": "Stop signal sent. The pipeline will halt after the current scene completes.",
    }


@app.get("/api/status", summary="Honest system & GPU tunnel status")
async def get_system_status():
    """
    Actively queries the backend state and pings the configured Google Colab
    GPU tunnel /health endpoint with ngrok-skip-browser-warning to report
    true connectivity, GPU device, and latency without false claims.
    """
    colab_info = await asyncio.to_thread(ping_colab_tunnel)
    return {
        "status": "online",
        "backend": "online",
        "pqc_model_loaded": weight_manager.is_pqc_loaded,
        "lora_adapter_available": weight_manager.has_lora_adapter,
        "device": weight_manager.device,
        "colab": colab_info,
        "supported_languages": ["English", "Hindi (Devanagari)", "Bengali (Bangla)"],
    }


@app.post("/api/config", summary="Update Colab ngrok tunnel URL")
async def update_colab_config(config: ColabConfigRequest):
    """Dynamically configure the active Google Colab ngrok tunnel URL and immediately test it."""
    url = set_colab_url(config.colab_url)
    ping_result = await asyncio.to_thread(ping_colab_tunnel, url)
    return {
        "status": "updated",
        "colab_url": url,
        "colab": ping_result,
    }


@app.post("/api/generate-scene", summary="Directly generate a single scene illustration")
async def generate_single_scene(request: SingleSceneRequest):
    """
    Directly renders a single scene illustration through the Colab GPU tunnel
    or high-fidelity local fallback, returning the image data URL directly.
    Ideal for serverless (Vercel) environments to avoid background task freeze.
    """
    use_quantum = request.model == "quantum"
    is_mock = request.model == "mock"
    try:
        clean_text = normalize_multilingual_text(request.scene_prompt)
        detected_lang = detect_language(clean_text)
        _, diag = weight_manager.enhance_features(clean_text, use_quantum=use_quantum)

        if is_mock:
            image_url = await asyncio.to_thread(
                generate_mock_scene_image, request.scene_prompt, request.scene_id, request.seed
            )
        else:
            image_url = await asyncio.to_thread(
                generate_scene_image, request.scene_prompt, request.scene_id, use_quantum, request.seed
            )
        return {
            "status": "success",
            "scene_id": request.scene_id,
            "seed": request.seed,
            "image_url": image_url,
            "quantum_diagnostics": diag,
            "detected_language": detected_lang,
            "model": request.model,
        }
    except Exception as exc:
        print(f"[ERR] Error generating scene {request.scene_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/compare-scene", summary="Side-by-side Quantum vs Standard comparison")
async def compare_scene(request: CompareSceneRequest):
    """
    Generates both Quantum-Enhanced and Standard SDXL scene illustrations
    using the same prompt to demonstrate the visual and Hilbert space differentiator.
    """
    try:
        clean_text = normalize_multilingual_text(request.scene_prompt)
        detected_lang = detect_language(clean_text)
        _, quantum_diag = weight_manager.enhance_features(clean_text, use_quantum=True)
        _, standard_diag = weight_manager.enhance_features(clean_text, use_quantum=False)

        quantum_img = await asyncio.to_thread(
            generate_scene_image, request.scene_prompt, request.scene_id, True, request.seed
        )
        standard_img = await asyncio.to_thread(
            generate_scene_image, request.scene_prompt, request.scene_id + 100, False, request.seed
        )

        return {
            "status": "success",
            "scene_prompt": request.scene_prompt,
            "seed": request.seed,
            "detected_language": detected_lang,
            "quantum": {
                "image_url": quantum_img,
                "diagnostics": quantum_diag,
            },
            "standard": {
                "image_url": standard_img,
                "diagnostics": standard_diag,
            }
        }
    except Exception as exc:
        print(f"[ERR] Error in comparison: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Serverless Proxy Route Aliases
# ---------------------------------------------------------------------------

@app.post("/api/generate", summary="Single scene generation proxy endpoint")
async def generate_proxy(request: SingleSceneRequest):
    """Serverless proxy route for generating a single storyboard frame."""
    return await generate_single_scene(request)
