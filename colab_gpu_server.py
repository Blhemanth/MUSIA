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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, 2048)
        residual = x
        normed = self.norm(x)
        compressed = self.compress(normed)

        orig_shape = compressed.shape
        flat = compressed.view(-1, self.n_qubits)

        q_out = torch.stack([self.quantum_layer(flat[i]) for i in range(flat.shape[0])])
        quantum_expectations = q_out.view(*orig_shape)

        expanded = self.expand(quantum_expectations)
        enhanced = residual + self.scale * expanded
        return enhanced


# ------------------------------------------------------------------------------
# 3. INITIALIZE MODELS
# ------------------------------------------------------------------------------
# Load PQC Model
pqc_enhancer = None
if os.path.exists(PQC_PATH):
    try:
        pqc_enhancer = QuantumFeatureEnhancer(embed_dim=2048, n_qubits=4, n_layers=2)
        state_dict = torch.load(PQC_PATH, map_location=DEVICE, weights_only=True)
        pqc_enhancer.load_state_dict(state_dict)
        pqc_enhancer.to(DEVICE).eval()
        print(f"[OK] Successfully loaded Quantum PQC Enhancer from {PQC_PATH}")
    except Exception as e:
        print(f"[WARN] Error loading PQC weights: {e}")
else:
    print(f"[INFO] PQC weights not found at {PQC_PATH}. Using fallback mode.")

# Load SDXL Base Pipeline
print("[MUSIA COLAB] Loading SDXL base model (stabilityai/stable-diffusion-xl-base-1.0)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    use_safetensors=True,
    variant="fp16" if DEVICE == "cuda" else None,
)
pipe = pipe.to(DEVICE)

if DEVICE == "cuda":
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

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
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "Cinematic storyboard illustration")
    use_quantum = data.get("use_quantum", True)
    scene_id = data.get("scene_id", 1)

    print(f"\n[GEN] Processing Scene #{scene_id} | Quantum: {use_quantum}")
    print(f"      Prompt: {prompt[:80]}...")

    try:
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
                # PQC operates in float32 then converts back to pipeline dtype
                orig_dtype = prompt_embeds.dtype
                prompt_embeds = pqc_enhancer(prompt_embeds.float()).to(orig_dtype)
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
        buf.seek(0)
        print(f"[OK] Scene #{scene_id} rendered successfully ({buf.getbuffer().nbytes} bytes)!")
        return send_file(buf, mimetype="image/png")

    except Exception as exc:
        print(f"[ERR] Generation error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------------------
# 5. ANTI-SLEEP HEARTBEAT THREAD & RUNNER
# ------------------------------------------------------------------------------
def keep_alive_worker():
    """Background heartbeat that prevents kernel idling."""
    while True:
        time.sleep(120)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[HEARTBEAT] GPU Active | Reserved VRAM: {vram_gb:.2f} GB")


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

    # Launch ngrok tunnel
    tunnel = ngrok.connect(PORT)
    public_url = tunnel.public_url

    print("\n" + "=" * 65)
    print("  🚀 MUSIA GOOGLE COLAB GPU BRIDGE ONLINE")
    print("=" * 65)
    print(f"  Public Tunnel URL: {public_url}")
    print(f"  Copy this URL to backend/generate_on_colab.py (line 25)")
    print(f'  Example: COLAB_NGROK_URL = "{public_url}"')
    print("=" * 65 + "\n")

    app.run(host="0.0.0.0", port=PORT)
