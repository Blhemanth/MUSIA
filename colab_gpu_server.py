"""
colab_gpu_server.py
MUSIA - Google Colab GPU Bridge Server
Supports SDXL + 4-Qubit Parameterized Quantum Circuit (PQC) Enhancer + LoRA

Configured for Google Drive Paths:
- SDXL LoRA Adapter: /content/drive/MyDrive/adapter_model.safetensors
- Quantum PQC Model: /content/drive/MyDrive/best_pqc_enhancer_fulldataset.pt
"""

# ==============================================================================
# CELL 1: DEPENDENCY INSTALLATION & GOOGLE DRIVE MOUNT
# ==============================================================================
# !pip install -q diffusers transformers accelerate pyngrok torch flask torchvision pillow pennylane
# from google.colab import drive
# drive.mount('/content/drive')

import io
import os
import sys
import time
import threading
import torch
import torch.nn as nn
from flask import Flask, request, send_file, jsonify
from diffusers import StableDiffusionXLPipeline
from pyngrok import ngrok

# Try importing PennyLane for quantum execution; fallback to PyTorch simulation if needed
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION
# ------------------------------------------------------------------------------
LORA_PATH = "/content/drive/MyDrive/adapter_model.safetensors"
PQC_PATH  = "/content/drive/MyDrive/best_pqc_enhancer_fulldataset.pt"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[MUSIA COLAB] Running on device: {DEVICE}")
if DEVICE != "cuda":
    print("[WARNING] GPU not detected! Go to: Runtime -> Change runtime type -> T4 GPU")

# ------------------------------------------------------------------------------
# 2. 4-QUBIT PARAMETERIZED QUANTUM CIRCUIT (PQC) ARCHITECTURE
# ------------------------------------------------------------------------------
def create_quantum_layer(n_qubits: int = 4, n_layers: int = 2):
    if HAS_PENNYLANE:
        try:
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch")
            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits))
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits)}
            return qml.qnn.TorchLayer(circuit, weight_shapes)
        except Exception as exc:
            print(f"[WARN] PennyLane init error: {exc}")

    class TorchQuantumLayer(nn.Module):
        def __init__(self, n_q: int, n_l: int):
            super().__init__()
            self.weights = nn.Parameter(torch.randn(n_l, n_q) * 0.1)

        def forward(self, x):
            phase = x.unsqueeze(-2) + self.weights
            z_exp = torch.cos(phase).mean(dim=-2)
            return torch.tanh(z_exp)

    return TorchQuantumLayer(n_qubits, n_layers)


class QuantumFeatureEnhancer(nn.Module):
    """Matches state dict: norm, compress, quantum_layer, expand, scale"""
    def __init__(self, embed_dim: int = 2048, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.norm = nn.LayerNorm(embed_dim)
        self.compress = nn.Linear(embed_dim, n_qubits)
        self.quantum_layer = create_quantum_layer(n_qubits, n_layers)
        self.expand = nn.Linear(n_qubits, embed_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        orig_dtype = x.dtype

        # Always evaluate quantum simulation strictly on CPU to avoid device mismatch with PennyLane default.qubit
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        residual = x_cpu
        normed = self.norm(x_cpu)
        compressed = self.compress(normed)

        orig_shape = compressed.shape
        flat = compressed.view(-1, self.n_qubits)

        q_list = []
        for i in range(flat.shape[0]):
            tok = flat[i].detach()
            tok_out = self.quantum_layer(tok)
            q_list.append(tok_out.detach().cpu())

        quantum_expectations = torch.stack(q_list).view(*orig_shape)
        expanded = self.expand(quantum_expectations)
        enhanced = residual + self.scale * expanded

        # Explicitly clean up intermediate CPU tensors
        del x_cpu, residual, normed, compressed, flat, q_list, quantum_expectations, expanded

        # Return back in exact device and dtype expected by SDXL
        return enhanced.to(device=orig_device, dtype=orig_dtype)


# ------------------------------------------------------------------------------
# 3. INITIALIZE MODELS
# ------------------------------------------------------------------------------
# Load PQC Model - STRICTLY kept on CPU to match PennyLane's default.qubit simulator
pqc_enhancer = None
if os.path.exists(PQC_PATH):
    try:
        pqc_enhancer = QuantumFeatureEnhancer(embed_dim=2048, n_qubits=4, n_layers=2)
        state_dict = torch.load(PQC_PATH, map_location="cpu", weights_only=True)
        pqc_enhancer.load_state_dict(state_dict)
        pqc_enhancer = pqc_enhancer.to("cpu")
        pqc_enhancer.eval()
        print(f"[OK] Successfully loaded Quantum PQC Enhancer from {PQC_PATH}")
    except Exception as e:
        print(f"[WARN] Error loading PQC weights: {e}")
else:
    print(f"[INFO] PQC weights not found at {PQC_PATH}. Using fallback mode.")

# Load SDXL Base Pipeline directly on GPU (T4 15GB VRAM) with memory-efficient attention & VAE tiling.
# Keeping SDXL on GPU instead of CPU offload protects Colab's 12.7GB System RAM from filling up!
print("[MUSIA COLAB] Loading SDXL base model (stabilityai/stable-diffusion-xl-base-1.0)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    use_safetensors=True,
    variant="fp16" if DEVICE == "cuda" else None,
)

if DEVICE == "cuda":
    pipe = pipe.to("cuda")
    # Slices cross-attention computation to eliminate VRAM spikes (<10GB peak on T4)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
    except Exception:
        try:
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except Exception:
            pass
else:
    pipe = pipe.to("cpu")

# Load LoRA Adapter from Google Drive
if os.path.exists(LORA_PATH):
    try:
        pipe.load_lora_weights(LORA_PATH)
        print(f"[OK] Successfully loaded SDXL LoRA adapter from {LORA_PATH}")
    except Exception as e:
        print(f"[WARN] Error loading LoRA adapter: {e}")
else:
    print(f"[INFO] LoRA adapter not found at {LORA_PATH}. Using base SDXL.")

print("[OK] All Neural & Quantum Models Ready!")

# ------------------------------------------------------------------------------
# 4. FLASK API SERVICE
# ------------------------------------------------------------------------------
app = Flask(__name__)

COLAB_SECRET_TOKEN = os.environ.get("COLAB_SECRET_TOKEN", "")

def verify_token():
    if not COLAB_SECRET_TOKEN:
        return True
    custom_token = request.headers.get("X-MUSIA-Token", "")
    auth_header = request.headers.get("Authorization", "")
    bearer_token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
    return (custom_token == COLAB_SECRET_TOKEN) or (bearer_token == COLAB_SECRET_TOKEN)

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "device": DEVICE,
        "pqc_loaded": pqc_enhancer is not None,
        "lora_loaded": os.path.exists(LORA_PATH),
    })

@app.route("/generate", methods=["POST"])
def generate():
    if not verify_token():
        return jsonify({"error": "Unauthorized: invalid or missing MUSIA secret token"}), 401
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "Cinematic storyboard illustration")
    use_quantum = data.get("use_quantum", True)
    scene_id = data.get("scene_id", 1)

    print(f"\n[GEN] Processing Scene #{scene_id} | Quantum: {use_quantum}")
    print(f"      Prompt: {prompt[:80]}...")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            # Encode prompt through SDXL dual text encoders
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompt,
                device=DEVICE,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

            # Apply Quantum Feature Enhancement if enabled
            if use_quantum and pqc_enhancer is not None:
                prompt_embeds = pqc_enhancer(prompt_embeds)
                print("      [PQC] Embeddings modulated in 4-qubit Hilbert space.")

            # Generate 16:9 cinematic storyboard illustration
            result = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=28,
                guidance_scale=7.5,
                height=576,
                width=1024,
            )
            image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG", quality=95)
        img_bytes = buf.getvalue()
        buf.close()

        # Aggressively release references & force garbage collection to keep System RAM low
        del result, image
        if 'prompt_embeds' in locals():
            del prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[OK] Scene #{scene_id} rendered successfully ({len(img_bytes)} bytes)!")
        return send_file(io.BytesIO(img_bytes), mimetype="image/png")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ERR] Generation error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------------------
# 5. ANTI-SLEEP HEARTBEAT THREAD & RUNNER
# ------------------------------------------------------------------------------
def keep_alive_worker():
    """Background heartbeat that prevents kernel idling and monitors RAM."""
    import psutil
    while True:
        time.sleep(60)
        sys_ram = psutil.virtual_memory()
        vram_str = ""
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            vram_str = f"| GPU VRAM: {vram_gb:.2f} GB"
        print(f"[HEARTBEAT] System RAM: {sys_ram.used / (1024**3):.2f}/{sys_ram.total / (1024**3):.2f} GB ({sys_ram.percent}%) {vram_str}")


if __name__ == "__main__":
    PORT = 8000

    # Start background anti-sleep thread
    heartbeat = threading.Thread(target=keep_alive_worker, daemon=True)
    heartbeat.start()

    # --------------------------------------------------------------------------
    # Paste your ngrok auth token here (from https://dashboard.ngrok.com/get-started/your-authtoken)
    # --------------------------------------------------------------------------
    NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"
    if NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTHTOKEN_HERE":
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Clean up any previous dangling ngrok sessions to avoid "Listener closed"
    try:
        ngrok.kill()
    except Exception:
        pass

    # Launch fresh ngrok tunnel
    tunnel = ngrok.connect(PORT)
    public_url = tunnel.public_url

    print("\n" + "=" * 65)
    print("  🚀 MUSIA GOOGLE COLAB GPU BRIDGE ONLINE")
    print("=" * 65)
    print(f"  Public Tunnel URL: {public_url}")
    print("=" * 65 + "\n")

    app.run(host="0.0.0.0", port=PORT, use_reloader=False)
