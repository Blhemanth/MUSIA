"""
backend/generate_on_colab.py
MUSIA - Multilingual Story Illustration
Google Colab Cloud GPU Bridge & Local Quantum Inference Integration

Connects:
- Active Google Colab ngrok tunnel (when online)
- Trained 4–8 Qubit Parameterized Quantum Circuit (`best_pqc_enhancer_fulldataset.pt`)
- Trained SDXL LoRA Attention Adapter (`adapter_model.safetensors`)
- Multilingual Typography & Scene Illustration Rendering (English, Hindi, Bengali)
"""

import os
import requests
from typing import Optional

from .quantum_pipeline import (
    render_multilingual_scene_illustration,
    weight_manager,
    OUTPUT_DIR,
)
from .story_parser import detect_language, normalize_multilingual_text

# Active Google Colab ngrok tunnel URL (can be overridden via COLAB_NGROK_URL env var or at runtime)
COLAB_NGROK_URL = os.environ.get("COLAB_NGROK_URL", "https://carnation-dislike-nervous.ngrok-free.dev")


def get_colab_url() -> str:
    """Return the currently configured Colab ngrok tunnel URL."""
    global COLAB_NGROK_URL
    return COLAB_NGROK_URL


def set_colab_url(new_url: str) -> str:
    """Dynamically update the active Colab ngrok tunnel URL."""
    global COLAB_NGROK_URL
    if new_url:
        COLAB_NGROK_URL = new_url.strip().rstrip("/")
    return COLAB_NGROK_URL


def ping_colab_tunnel(url: Optional[str] = None, timeout: float = 3.5) -> dict:
    """
    Pings the Colab GPU tunnel /health endpoint with ngrok-skip-browser-warning.
    Returns status dict with latency, device info, and error message if offline.
    """
    target = (url or COLAB_NGROK_URL).strip().rstrip("/")
    if not target or target.startswith("https://your-"):
        return {
            "configured": False,
            "url": target,
            "online": False,
            "error": "Colab ngrok URL not configured."
        }

    import time
    start_t = time.time()
    colab_secret = os.environ.get("COLAB_SECRET_TOKEN", "")
    headers = {
        "ngrok-skip-browser-warning": "true",
        "User-Agent": "MUSIA-Client/3.0",
        "Accept": "application/json",
    }
    if colab_secret:
        headers["X-MUSIA-Token"] = colab_secret
        headers["Authorization"] = f"Bearer {colab_secret}"
    try:
        health_url = f"{target}/health"
        resp = requests.get(health_url, headers=headers, timeout=timeout)
        latency_ms = int((time.time() - start_t) * 1000)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
            return {
                "configured": True,
                "url": target,
                "online": True,
                "latency_ms": latency_ms,
                "device": data.get("device", "cuda"),
                "pqc_loaded": data.get("pqc_loaded", True),
                "lora_loaded": data.get("lora_loaded", True),
                "error": None
            }
        else:
            return {
                "configured": True,
                "url": target,
                "online": False,
                "latency_ms": latency_ms,
                "error": f"Colab returned HTTP {resp.status_code}"
            }
    except Exception as exc:
        return {
            "configured": True,
            "url": target,
            "online": False,
            "latency_ms": int((time.time() - start_t) * 1000),
            "error": f"Colab tunnel offline or unreachable: {str(exc)}"
        }


def generate_scene_image(scene_prompt: str, scene_id: int, use_quantum: bool = True, seed: Optional[int] = None) -> str:
    """
    Executes the scene illustration generation pipeline utilizing the trained
    model weights (Standard SDXL and Quantum-Enhanced 4-8 qubit PQC models):
    1. If the live Colab GPU tunnel is reachable, dispatches the prompt
       and quantum parameters to the cloud GPU.
    2. If the tunnel is offline or in local environment, executes the local
       quantum latent pipeline with the trained PQC enhancer (`best_pqc_enhancer_fulldataset.pt`)
       and renders high-fidelity multilingual cinematic frames in Hindi, Bengali, Tamil, Kannada, or English.

    Args:
        scene_prompt: Cinematic prompt text for this scene.
        scene_id: 1-based sequential scene index.
        use_quantum: True for Quantum-Enhanced PQC model, False for Standard SDXL.
        seed: Optional RNG seed for deterministic frame recreation.

    Returns:
        Relative URL string or data URL.
    """
    print(f"[SCENE] Processing scene {scene_id} | Quantum: {use_quantum} | Seed: {seed} | Model weights: PQC={weight_manager.is_pqc_loaded}")
    return render_multilingual_scene_illustration(
        scene_prompt=scene_prompt,
        scene_id=scene_id,
        use_quantum=use_quantum,
        colab_url=get_colab_url(),
        seed=seed,
    )


def generate_mock_scene_image(scene_prompt: str, scene_id: int, seed: Optional[int] = None) -> str:
    """
    Rapid preview illustration generator supporting English, Hindi, Bengali,
    Tamil, and Kannada text. Generates an instant preview without GPU delays.

    Args:
        scene_prompt: Cinematic prompt string for this scene.
        scene_id: 1-based scene index.
        seed: Optional RNG seed.

    Returns:
        Relative HTTP URL or data URL.
    """
    return render_multilingual_scene_illustration(
        scene_prompt=scene_prompt,
        scene_id=scene_id,
        use_quantum=False,
        colab_url=None,  # Force local preview rendering
        seed=seed,
    )
