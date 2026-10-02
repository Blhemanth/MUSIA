"""
colab_gpu_server.py
MUSIA - Google Colab GPU Bridge Server
Run this script inside a Google Colab notebook with a GPU runtime (T4 or A100).

Setup instructions in Colab:
---------------------------------------------
# Cell 1: Install dependencies
!pip install -q diffusers transformers accelerate pyngrok torch flask torchvision pillow

# Cell 2: Run server
# (Copy-paste this file's contents into a cell and execute)
---------------------------------------------
"""

import io
import os
import sys
import torch
from flask import Flask, request, send_file, jsonify
from diffusers import StableDiffusionXLPipeline
from pyngrok import ngrok

# 1. Initialize Flask app
app = Flask(__name__)

# 2. Check Device & Load Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[MUSIA COLAB] Initializing GPU pipeline on device: {DEVICE}")

if DEVICE != "cuda":
    print("[WARN] No GPU detected! Please navigate to Runtime -> Change runtime type -> T4 GPU in Colab.")

print("[MUSIA COLAB] Loading SDXL pipeline (stabilityai/stable-diffusion-xl-base-1.0)...")
try:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if DEVICE == "cuda" else None,
    )
    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()
    print("[MUSIA COLAB] SDXL Pipeline successfully loaded and optimized!")
except Exception as e:
    print(f"[ERR] Failed to load SDXL pipeline: {e}")
    pipe = None


# 3. Endpoints
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "MUSIA Colab GPU Worker",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": pipe is not None,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Accepts JSON:
    {
        "prompt": "Cinematic scene prompt...",
        "use_quantum": true,
        "scene_id": 1
    }
    Returns: PNG image stream
    """
    if pipe is None:
        return jsonify({"error": "SDXL model not loaded"}), 500

    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "Cinematic masterpiece storyboard illustration")
    use_quantum = data.get("use_quantum", True)
    scene_id = data.get("scene_id", 1)

    print(f"\n[GEN] Processing Scene #{scene_id}")
    print(f"      Quantum Mode: {use_quantum}")
    print(f"      Prompt: {prompt[:100]}...")

    try:
        with torch.inference_mode():
            # 16:9 cinematic aspect ratio (1024x576)
            result = pipe(
                prompt=prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=576,
                width=1024,
            )
            image = result.images[0]

        # Convert to in-memory PNG buffer
        buf = io.BytesIO()
        image.save(buf, format="PNG", quality=95)
        buf.seek(0)
        print(f"[OK] Scene #{scene_id} generated successfully!")
        return send_file(buf, mimetype="image/png")

    except Exception as exc:
        print(f"[ERR] Generation error: {exc}")
        return jsonify({"error": str(exc)}), 500


# 4. Ngrok Tunnel & Runner
if __name__ == "__main__":
    PORT = 8000

    # If you have an ngrok auth token:
    # ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

    # Connect ngrok to local port 8000
    tunnel = ngrok.connect(PORT)
    public_url = tunnel.public_url

    print("\n" + "=" * 65)
    print("  MUSIA COLAB GPU BRIDGE IS READY")
    print("=" * 65)
    print(f"  Public Tunnel URL: {public_url}")
    print("  Paste this URL into backend/generate_on_colab.py (line 25)")
    print(f'  Example: COLAB_NGROK_URL = "{public_url}"')
    print("=" * 65 + "\n")

    # Listen on 0.0.0.0 to prevent IPv4/IPv6 localhost binding issues
    app.run(host="0.0.0.0", port=PORT)
