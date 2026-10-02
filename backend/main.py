"""
backend/main.py
MUSIA - Multilingual Story Illustration
Phase 2+: FastAPI Backend Server — Multilingual Processing (English, Hindi, Bengali)
           with Quantum-Enhanced PQC Inference Pipeline
"""

import os
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from .story_parser import (
    parse_story_to_scenes,
    parse_story_scenes_detailed,
    detect_language,
    normalize_multilingual_text,
)
from .generate_on_colab import generate_scene_image, generate_mock_scene_image
from .quantum_pipeline import weight_manager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    story: str

    @field_validator("story")
    @classmethod
    def story_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("The 'story' field must not be empty.")
        return v.strip()


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


@app.get("/api/health", summary="Health check")
async def health_check():
    """Health check confirming API status and model weight availability."""
    return {
        "status": "success",
        "message": "MUSIA API v3 is up and running.",
        "pqc_model_loaded": weight_manager.is_pqc_loaded,
        "lora_adapter_available": weight_manager.has_lora_adapter,
        "device": weight_manager.device,
        "supported_languages": ["English", "Hindi (Devanagari)", "Bengali (Bangla)"],
    }


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
    """
    details = parse_story_scenes_detailed(request.story)
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
