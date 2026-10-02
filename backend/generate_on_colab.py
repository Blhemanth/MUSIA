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

# Active Google Colab ngrok tunnel URL (can be overridden via COLAB_NGROK_URL env var)
COLAB_NGROK_URL = os.environ.get("COLAB_NGROK_URL", "https://lend-eggplant-majesty.ngrok-free.dev")


def generate_scene_image(scene_prompt: str, scene_id: int, use_quantum: bool = True) -> str:
    """
    Executes the scene illustration generation pipeline utilizing the trained
    model weights (Standard SDXL and Quantum-Enhanced 4-8 qubit PQC models):
    1. If the live Colab GPU tunnel is reachable, dispatches the prompt
       and quantum parameters to the cloud GPU.
    2. If the tunnel is offline or in local environment, executes the local
       quantum latent pipeline with the trained PQC enhancer (`best_pqc_enhancer_fulldataset.pt`)
       and renders high-fidelity multilingual cinematic frames in Hindi, Bengali, or English.

    Args:
        scene_prompt: Cinematic prompt text for this scene.
        scene_id: 1-based sequential scene index.
        use_quantum: True for Quantum-Enhanced PQC model, False for Standard SDXL.

    Returns:
        Relative URL string (e.g. '/static/generated_scenes/scene_1.png')
    """
    print(f"[SCENE] Processing scene {scene_id} | Quantum: {use_quantum} | Model weights: PQC={weight_manager.is_pqc_loaded}")
    return render_multilingual_scene_illustration(
        scene_prompt=scene_prompt,
        scene_id=scene_id,
        use_quantum=use_quantum,
        colab_url=COLAB_NGROK_URL,
    )


def generate_mock_scene_image(scene_prompt: str, scene_id: int) -> str:
    """
    Rapid preview illustration generator supporting English, Hindi, and Bengali
    text. Generates an instant preview without GPU delays.

    Args:
        scene_prompt: Cinematic prompt string for this scene.
        scene_id: 1-based scene index.

    Returns:
        Relative HTTP URL: '/static/generated_scenes/scene_<scene_id>.png'
    """
    return render_multilingual_scene_illustration(
        scene_prompt=scene_prompt,
        scene_id=scene_id,
        use_quantum=False,
        colab_url=None,  # Force local preview rendering
    )
