# ⚛️ MUSIA — Complete Learning Guide
### *End-to-End Technical Deep Dive: From Story to Cinematic Storyboard*

> This guide explains **every layer** of the MUSIA project — from what it is, how a story gets broken into scenes, how the frontend talks to Google Colab, how the quantum model works, and how SDXL generates the final images.

---

## 📖 Table of Contents

1. [What is MUSIA?](#1-what-is-musia)
2. [How Stories are Broken into Scenes](#2-how-stories-are-broken-into-scenes)
3. [How the Frontend Communicates with Google Colab](#3-how-the-frontend-communicates-with-google-colab)
4. [What is the Colab Code and How it Works](#4-what-is-the-colab-code-and-how-it-works)
5. [Hindi / Bengali Story Handling — Translation or Direct?](#5-hindi--bengali-story-handling--translation-or-direct)
6. [How the Quantum Model Works](#6-how-the-quantum-model-works)
7. [Frontend & Backend — What They Are and How They Connect](#7-frontend--backend--what-they-are-and-how-they-connect)
8. [How the SDXL Model Works](#8-how-the-sdxl-model-works)
9. [Full Project Architecture](#9-full-project-architecture)

---

## 1. What is MUSIA?

**MUSIA** stands for **Multilingual Story Illustration**. It is an AI-powered web application that:

- **Takes a written story** (a novel, screenplay, short story) as input — in **English, Hindi, or Bengali**
- **Breaks it into cinematic scenes** automatically using an intelligent story parser
- **Generates a high-fidelity image** for each scene using a combination of:
  - A **Parameterized Quantum Circuit (PQC)** that enhances the text prompt embeddings
  - **Stable Diffusion XL (SDXL)** — a state-of-the-art image generation model
  - A **LoRA adapter** (`adapter_model.safetensors`) that fine-tunes SDXL for cinematic storyboard aesthetics
- **Displays the results** in a beautiful "Quantum Studio" dark-mode web interface with glassmorphism, neon glows, and real-time progress tracking

### In Simple Words:
```
You paste a story  →  MUSIA splits it into scenes  →  Each scene gets a cinematic image
```

### Three Generation Modes:
| Mode | What Happens | Speed |
|---|---|---|
| **Quantum-Enhanced SDXL** | Runs full PQC → SDXL on Colab GPU | Slowest (best quality) |
| **Standard SDXL** | Runs plain SDXL on Colab GPU (no quantum) | Medium |
| **Fast Local Preview (Mock)** | Renders a local cinematic placeholder image using PIL | Instant |

---

## 2. How Stories are Broken into Scenes

**File:** `backend/story_parser.py`

This is the "brain" that reads your story and divides it into separate visual scenes. Here is exactly how it works, step by step:

### Step 1 — Unicode Normalization
```python
def normalize_multilingual_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")
```
- Converts the text to **NFC Unicode** — this is critical for Hindi and Bengali because their characters (matras, conjuncts) can be represented in multiple Unicode forms. NFC ensures they are all in one consistent canonical form.
- Normalizes Windows-style line endings (`\r\n`) to Unix (`\n`).

### Step 2 — Language Detection
```python
def detect_language(text: str) -> str:
    # Checks Unicode codepoints:
    # Devanagari (Hindi):  U+0900 – U+097F
    # Bengali:             U+0980 – U+09FF
    # Tamil:               U+0B80 – U+0BFF
    # Kannada:             U+0C80 – U+0CFF
    # English (Latin):     A-Z, a-z
```
- Counts characters per script
- If ≥50% of characters belong to one script → that language is declared the primary language
- Returns `'hi'` (Hindi), `'bn'` (Bengali), `'en'` (English), or `'mixed'`

### Step 3 — Choosing a Segmentation Strategy (Priority Order)

The parser tries three strategies in priority order:

#### Strategy 1: Chapter/Part Markers
If the text contains headers like `Chapter 1:`, `अध्याय`, `অধ্যায়`, `Scene`, `Part`, etc., it splits at those boundaries.

```python
_CHAPTER_PATTERN = re.compile(
    r"^\s*(?:(?:chapter|part|section|scene|act|prologue|epilogue)\b[^\n:]*|"
    r"(?:अध्याय|भाग|दृश्य|खंड|अंक|प्रस्तावना|उपसंहार)[^\n:]*|"
    r"(?:অধ্যায়|পর্ব|বিভাগ|দৃশ্য|অঙ্ক|ভূমিকা|উপসংহার)[^\n:]*)..."
)
```

This works for English, Hindi, and Bengali chapter markers simultaneously.

#### Strategy 2: Paragraph Breaks
If the text has `\n\n` (blank lines) separating paragraphs, those paragraphs become scenes.

#### Strategy 3: Sentence Chunking (Default)
If no chapters or paragraphs are detected, the text is split into **individual sentences** (supports `.`, `!`, `?`, `।`, `॥`) and then grouped into chunks of **3 sentences per scene**.

```python
split_pattern = r'(?:(?<=[।॥!?])\s+)|(?:(?<=\.)\s+)'
```
- `।` is the **Purna Viram** — the full stop in Hindi and Bengali
- `॥` is the **Double Purna Viram** — used at the end of verses

Abbreviations like `Dr.`, `Mr.`, `डॉ.`, `ड.` are **protected** so the parser doesn't split them mid-word.

### Step 4 — Formatting Scene Prompts
Each chunk gets formatted as a cinematic prompt:
```python
prompt = f"Cinematic story illustration, scene {idx}: {chunk}"
```

### Example:
```
Input:
  "A knight stands at the gate. The castle looms in darkness. Thunder rolls across the sky."

Output:
  ["Cinematic story illustration, scene 1: A knight stands at the gate.
    The castle looms in darkness. Thunder rolls across the sky."]
```

### Cap at 12 Scenes
The backend enforces a max of **12 scenes** (`MAX_SCENES = 12`) to prevent excessive GPU usage and UI overload.

---

## 3. How the Frontend Communicates with Google Colab

This is the most important architectural piece to understand. Here's the full flow:

```
Browser (User's computer)
    ↓  HTTP POST  (to FastAPI backend at port 8000)
FastAPI Backend (local or Vercel)
    ↓  HTTP POST  (to ngrok public URL)
ngrok Tunnel
    ↓  forwards to
Google Colab (Flask server running on GPU)
    ↓  returns PNG image bytes
FastAPI Backend receives image → returns to Browser
```

### Why ngrok?
Google Colab runs in Google's cloud data centers. It has **no public IP address** by default. **ngrok** creates a secure public HTTPS tunnel that exposes the Colab Flask server to the internet. This allows the local FastAPI backend to reach it.

### Step-by-Step Communication Flow:

#### Step 1 — User Pastes Story & Clicks "Parse Story"
The frontend sends:
```javascript
fetch('/api/parse-story', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ story: userStoryText })
})
```
The FastAPI backend runs `parse_story_to_scenes()` and returns a list of scene prompts. The frontend renders them as cards in the storyboard timeline.

#### Step 2 — User Selects Model & Clicks "Launch Pipeline"
```javascript
fetch('/api/start-generation', {
    method: 'POST',
    body: JSON.stringify({ model: 'quantum' })  // or 'standard' or 'mock'
})
```
This triggers the **background generation pipeline** in FastAPI. The endpoint returns immediately with `{"status": "started"}` — it does NOT wait for all images.

#### Step 3 — Frontend Polls for Progress
The frontend repeatedly calls the status endpoint every few seconds:
```javascript
setInterval(async () => {
    const status = await fetch('/api/generation-status').then(r => r.json());
    // status.current_scene  → which scene is being processed
    // status.image_urls     → array of completed image URLs/data URLs
    // status.is_running     → true/false
}, 3000);
```

#### Step 4 — FastAPI Sends Each Scene to Colab
For each scene in the pipeline, `generate_on_colab.py` is called:
```python
# 1. First pings the Colab /health endpoint (3 second timeout)
ping_resp = requests.get(f"{colab_url}/health", headers=headers, timeout=3.0)

# 2. If alive, POST the prompt to Colab /generate
resp = requests.post(colab_url + "/generate", json={
    "prompt": scene_prompt,
    "use_quantum": True,
    "scene_id": scene_id
}, timeout=120)  # 120 second timeout for SDXL generation

# 3. Colab returns raw PNG bytes
# 4. Backend base64-encodes them and sends back as data URL
```

#### Step 5 — Frontend Displays Images
As `image_urls` array grows in the polling response, the frontend renders each image in the storyboard gallery in real-time.

### Security in the Communication:
| Security Measure | How it Works |
|---|---|
| `X-MUSIA-Token` header | Shared secret token sent with every Colab request to prevent unauthorized GPU use |
| `Authorization: Bearer <token>` | Standard OAuth-style header as backup authentication |
| `ngrok-skip-browser-warning` header | Prevents ngrok's interstitial warning page from blocking API responses |
| Rate limiting | 120 requests/minute per IP enforced in FastAPI middleware |

---

## 4. What is the Colab Code and How it Works

**File:** `colab_gpu_server.py`

This script runs **inside Google Colab** and acts as a GPU-powered server. It is a complete standalone server.

### What Runs in Colab:

#### 1. Model Loading (runs at startup, once)
```python
# Load PQC Quantum Enhancer from Google Drive (KEPT ON CPU)
pqc_enhancer = QuantumFeatureEnhancer(embed_dim=2048, n_qubits=4, n_layers=2)
pqc_enhancer.load_state_dict(torch.load(PQC_PATH, map_location="cpu"))
pqc_enhancer.to("cpu").eval()

# Load SDXL base model on GPU (T4 / A100)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,  # Half precision to save VRAM
)
pipe.to("cuda")
pipe.enable_attention_slicing()   # Reduces VRAM spikes
pipe.enable_vae_tiling()          # Handles large images efficiently

# Load LoRA adapter for cinematic style
pipe.load_lora_weights("/content/drive/MyDrive/adapter_model.safetensors")
```

**Key design decision:** The PQC Enhancer is kept on **CPU** (even in Colab), not GPU. This is because PennyLane's `default.qubit` quantum simulator runs on CPU. Moving it to GPU would cause device mismatch errors. The SDXL model runs on **CUDA GPU** for speed.

#### 2. Flask API (the server that FastAPI talks to)
```python
@app.route("/health")
def health():
    return jsonify({"status": "online", "device": DEVICE, "pqc_loaded": True})

@app.route("/generate", methods=["POST"])
def generate():
    prompt = data["prompt"]
    use_quantum = data["use_quantum"]

    # 1. Encode text into SDXL embeddings (on GPU)
    prompt_embeds, negative_embeds, pooled_embeds, negative_pooled = pipe.encode_prompt(...)

    # 2. If quantum mode: pass embeddings through the PQC (on CPU)
    if use_quantum and pqc_enhancer:
        prompt_embeds = pqc_enhancer(prompt_embeds)
        print("[PQC] Embeddings modulated in 4-qubit Hilbert space.")

    # 3. Generate image with SDXL
    result = pipe(
        prompt_embeds=prompt_embeds,
        num_inference_steps=28,
        guidance_scale=7.5,
        height=576, width=1024  # 16:9 cinematic ratio
    )

    # 4. Return as PNG bytes
    image.save(buf, format="PNG")
    return send_file(buf, mimetype="image/png")
```

#### 3. ngrok Tunnel (exposes Colab to the internet)
```python
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN_HERE")
tunnel = ngrok.connect(PORT)
print(f"Public Tunnel URL: {tunnel.public_url}")
# e.g., "https://abc-def-123.ngrok-free.dev"
```
This URL is what you copy and paste into `COLAB_NGROK_URL` in your local `.env` file so the FastAPI backend can find Colab.

#### 4. Anti-Sleep Heartbeat Thread
```python
def keep_alive_worker():
    while True:
        time.sleep(60)
        print(f"[HEARTBEAT] RAM: {sys_ram.used}GB | VRAM: {vram_gb:.2f}GB")
```
Google Colab kills idle kernels. This heartbeat keeps the session alive by printing to stdout every 60 seconds.

### How to Set Up Colab (Step by Step):
1. Open Google Colab → Runtime → Change runtime type → **T4 GPU** (free) or A100 (Colab Pro)
2. Mount Google Drive: `drive.mount('/content/drive')`
3. Upload `adapter_model.safetensors` and `best_pqc_enhancer_fulldataset.pt` to Google Drive root
4. Install dependencies:
   ```bash
   !pip install diffusers transformers accelerate pyngrok torch flask torchvision pillow pennylane
   ```
5. Run `colab_gpu_server.py` — it loads models and prints the public ngrok URL
6. Copy the URL → paste into your `.env` as `COLAB_NGROK_URL=https://xxx.ngrok-free.dev`

---

## 5. Hindi / Bengali Story Handling — Translation or Direct?

> **The story is NOT translated to English. It is fed DIRECTLY into the model.**

This is a critical design choice. Here is exactly what happens:

### Flow for Hindi/Bengali Input:
```
Hindi/Bengali Story Text
    ↓
story_parser.py: normalize_multilingual_text()
    - NFC Unicode normalization
    - Preserves ZWJ/ZWNJ (required for Indic script rendering)
    ↓
detect_language() → returns 'hi' or 'bn'
    ↓
split_sentences_multilingual()
    - Splits on Purna Viram '।' and '॥' (not just '.' '!' '?')
    - Protects Indic abbreviations like डॉ. ड. मो.
    ↓
Scene Prompts (still in Hindi/Bengali):
  "Cinematic story illustration, scene 1: एक समय की बात है, राजा विक्रमादित्य के दरबार में..."
    ↓
compute_text_embedding() in quantum_pipeline.py
    - Tokenizes the Indic text using Unicode-aware regex
    - Applies language-specific bias to the embedding:
      lang_bias = {"hi": 0.35, "bn": 0.55, "en": 0.15}
    - Generates a 2048-dimensional latent vector
    ↓
PQC Enhancer → sent to SDXL
```

### What Does SDXL Do with Hindi/Bengali Text?

SDXL's text encoder (CLIP) was primarily trained on English. When given Hindi or Bengali text:
- CLIP tokenizes the text using byte-pair encoding (BPE) — Indic characters get encoded into subword tokens
- The semantic understanding is **less precise** than English, but MUSIA partially compensates via:
  1. **Language-aware token harmonics** — adds script-specific frequency modulations to the synthetic embeddings
  2. **Quantum PQC modulation** — enriches the embedding in Hilbert space regardless of language
  3. **The scene text** still carries visual context clues (even in Hindi/Bengali, scene descriptions contain visual nouns and verbs)

### Why Not Translate?
- Translation would lose cultural nuance, names, places, and specific visual elements unique to the story
- It adds a dependency on a translation API or model (latency, cost, quality issues)
- The system is designed to be **multilingual-native**, not translation-dependent
- SDXL still generates visually coherent images from the structural patterns in the embeddings

### Language-Specific Features:
The system tells the user what language was detected:
```json
{
  "detected_language": "hi",
  "language_name": "Hindi (Devanagari)"
}
```
And on the rendered image itself, there's a language badge:
- `SCRIPT: हिन्दी (Hindi)` — teal color
- `SCRIPT: বাংলা (Bengali)` — cyan color
- `SCRIPT: English` — indigo color

---

## 6. How the Quantum Model Works

**File:** `backend/quantum_pipeline.py`

This is the most innovative part of MUSIA. Let's break it down completely.

### What Problem Does it Solve?

Standard text encoders like CLIP map text into Euclidean vector spaces — flat, linear representations. A **Parameterized Quantum Circuit (PQC)** operates in **Hilbert space** — a much richer mathematical space where:
- Quantum **superposition** allows a qubit to represent multiple states simultaneously
- Quantum **entanglement** creates non-linear correlations between dimensions that classical networks cannot replicate as efficiently
- This improves the expressiveness of prompt embeddings

### The Architecture (Layer by Layer):

```
Input Text Prompt
    ↓
compute_text_embedding()
    → Tokenize text (Indic-aware Unicode regex)
    → Hash each token → base value in [-0.5, +0.5]
    → Build sinusoidal harmonics: sin(freq * position + hash_val)
    → Shape: [1, seq_len, 2048]  (batch=1, ≤77 tokens, dim=2048)
    ↓
QuantumFeatureEnhancer.forward(x)
    ├── LayerNorm(2048)           → Stabilize input scale
    ├── Linear(2048 → 4)         → Compress to n_qubits=4 dimensions
    │       (Projects each token from 2048D to 4D for quantum input)
    ├── PennyLane Quantum Circuit (AngleEmbedding + BasicEntanglerLayers)
    │       → Each token's 4-dim vector processed through quantum gates
    │       → Output: 4 expectation values (one per qubit)
    ├── Linear(4 → 2048)         → Expand back to 2048 dimensions
    └── Residual + Scale:  x_enhanced = x_original + scale * expanded
    ↓
Output: Enhanced [1, seq_len, 2048] — quantum-modulated embeddings
```

### The Quantum Circuit Itself:

```python
@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    # Step 1: AngleEmbedding
    # Each of the 4 values in 'inputs' rotates one qubit by an angle:
    # qubit_0 ← RY(inputs[0])
    # qubit_1 ← RY(inputs[1])
    # qubit_2 ← RY(inputs[2])
    # qubit_3 ← RY(inputs[3])
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Step 2: BasicEntanglerLayers (2 layers)
    # Each layer applies:
    # - RX rotation on each qubit (parameterized by learned weights)
    # - CNOT gates between adjacent qubits to create entanglement
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # Step 3: Measurement
    # Returns Pauli-Z expectation value for each qubit: E[<Z_i>] ∈ [-1, +1]
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

### What is `AngleEmbedding`?

It rotates each qubit by an angle equal to the compressed feature value:
- `qubit_i` is set to `|ψ_i⟩ = cos(θ_i/2)|0⟩ + sin(θ_i/2)|1⟩`
- Where `θ_i = inputs[i]` (the 4-dimensional compressed text feature)

This **encodes classical text representation into quantum state space** (the Bloch sphere).

### What is `BasicEntanglerLayers`?

Two layers of:
1. **Parameterized rotations** — `RX(w_i)` on each qubit, where `w_i` are **learned weights** stored in `best_pqc_enhancer_fulldataset.pt`
2. **CNOT gates** — `CNOT(qubit_i, qubit_{i+1})` creating entanglement between adjacent qubits

This is the quantum equivalent of a neural network layer. The weights are **trained** (not random) — they were learned from a dataset of story prompts to optimally modulate embeddings.

### The Expectation Values

After measurement, each qubit returns its **PauliZ expectation value**:
- `+1.0` → qubit is likely in state `|0⟩`
- `-1.0` → qubit is likely in state `|1⟩`
- Values in between represent **quantum superposition**

These 4 expectation values `[q0, q1, q2, q3]` are a quantum "fingerprint" of the scene prompt. They are expanded back to 2048 dimensions and added as a residual correction to the original embedding.

### The Trained Weights:

**File:** `best_pqc_enhancer_fulldataset.pt` (94 KB PyTorch state dict)

Contains:
| Key | Shape | Purpose |
|---|---|---|
| `norm.weight` | [2048] | LayerNorm scale |
| `norm.bias` | [2048] | LayerNorm shift |
| `compress.weight` | [4, 2048] | Projects 2048D → 4D |
| `compress.bias` | [4] | Compress bias |
| `quantum_layer.weights` | [2, 4] | PennyLane variational angles (2 layers × 4 qubits) |
| `expand.weight` | [2048, 4] | Projects 4D → 2048D |
| `expand.bias` | [2048] | Expand bias |
| `scale` | scalar | Residual strength |

### Quantum Diagnostics in API Response:
```json
{
  "quantum_diagnostics": {
    "mode": "quantum_enhanced",
    "n_qubits": 4,
    "qubit_expectations": [0.4231, -0.3812, 0.6104, -0.1507],
    "quantum_scale": 0.9832,
    "status": "Quantum Hilbert Space Modulation Applied"
  }
}
```

These values are also rendered on the generated mock images:
```
PQC Qubits [q0..q3]: q0:+0.42  q1:-0.38  q2:+0.61  q3:-0.15
```

### PyTorch Fallback (When PennyLane is Unavailable):
```python
class TorchQuantumLayer(nn.Module):
    def forward(self, x):
        phase = x.unsqueeze(-2) + self.weights
        z_exp = torch.cos(phase).mean(dim=-2)
        return torch.tanh(z_exp)
```
This simulates quantum gate behavior using trigonometric functions — `cos` approximating PauliZ measurements, `tanh` approximating bounded expectation values. Not real quantum, but a mathematically inspired approximation that preserves the architectural behavior.

---

## 7. Frontend & Backend — What They Are and How They Connect

### The Frontend

**File:** `frontend/index.html` (188 KB, ~3548 lines)

A **Single Page Application (SPA)** built in HTML + Vanilla CSS + JavaScript (all in one file):

#### Visual Design System:
- **Color palette:** Neon cyan (`#00f2fe`), quantum violet (`#7928ca`), deep space (`#040714`)
- **Glassmorphism cards:** `backdrop-filter: blur(16px)` with semi-transparent borders
- **Fonts:** Plus Jakarta Sans (body), JetBrains Mono (code/metrics), Cinzel (headings)
- **Animations:** Floating, glow-pulse, shimmer-beam, quantum-spin
- **Quantum Canvas:** HTML5 Canvas with real-time particle physics — particles represent qubits, connected by entanglement lines that react to mouse movement

#### Libraries Loaded:
| Library | CDN Source | Purpose |
|---|---|---|
| `three.js` | jsdelivr | WebGL shader effects |
| `JSZip` | jsdelivr | One-click ZIP export of all images |
| `Tailwind CSS` | cdn.tailwindcss.com | Utility layout classes |
| Google Fonts | fonts.googleapis.com | Plus Jakarta Sans, JetBrains Mono, Cinzel |

#### Key UI Sections:
1. **Hero Header** — MUSIA title with quantum particle background
2. **Story Input Panel** — Textarea (max 5000 chars), language auto-detected on parse
3. **Model Selector** — Dropdown: Quantum-Enhanced SDXL / Standard SDXL / Fast Mock
4. **Control Buttons** — Parse Story, Launch Pipeline, Stop, Export ZIP
5. **Storyboard Timeline** — Scene cards rendered after parsing
6. **Generation Gallery** — Image cards that populate in real-time as scenes complete
7. **Quantum Diagnostics Panel** — Live qubit expectation values, tunnel status
8. **Status Bar** — Current scene number, total progress, model type, elapsed time

### The Backend

**File:** `backend/main.py` (FastAPI server, 624 lines)

A **FastAPI Python application** that:
- Serves `frontend/index.html` at route `/`
- Serves generated scene images at `/static/generated_scenes/`
- Exposes all REST API endpoints
- Manages **global generation state** (dict tracking current scene, image URLs, running flag)
- Runs image generation as `asyncio` background tasks (non-blocking)
- Applies security middleware (rate limiting, security headers, input sanitization)

### All API Endpoints:
| Endpoint | Method | What It Does |
|---|---|---|
| `/` | `GET` | Serves `frontend/index.html` |
| `/api/health` | `GET` | Backend + Colab tunnel + model status |
| `/api/status` | `GET` | Detailed system status with GPU ping & latency |
| `/api/parse-story` | `POST` | Splits story into ≤12 cinematic scenes |
| `/api/parse-story-detailed` | `POST` | Scenes + language detection metadata + token counts |
| `/api/start-generation` | `POST` | Starts background image generation pipeline |
| `/api/generation-status` | `GET` | Polling: current scene, image URLs, running state |
| `/api/stop-generation` | `POST` | Gracefully stops the pipeline after current scene |
| `/api/generate-scene` | `POST` | Single scene generation with quantum diagnostics |
| `/api/generate` | `POST` | Proxy alias for generate-scene (Vercel serverless) |
| `/api/compare-scene` | `POST` | Generates Quantum vs Standard side-by-side images |
| `/api/config` | `POST` | Update ngrok Colab tunnel URL at runtime |
| `/api/model-info` | `GET` | PQC architecture details + LoRA adapter info |
| `/static/generated_scenes/*` | `GET` | Serve saved scene PNG files |

### The Connection is "Polling-Based":

The frontend does **not** use WebSockets. Instead:
1. Sends `POST /api/start-generation` → backend returns `"started"` immediately
2. Backend runs generation in `asyncio` background task (non-blocking, event loop stays free)
3. Frontend polls `GET /api/generation-status` every few seconds
4. As each scene completes, its image URL appears in the status → frontend renders it

### Vercel Deployment:
`vercel.json` routes all `/api/*` requests to `api/index.py` which mounts the FastAPI app as a serverless function. Environment variables (`COLAB_NGROK_URL`, `COLAB_SECRET_TOKEN`) are set in Vercel Project Settings.

---

## 8. How the SDXL Model Works

### What is SDXL?

**Stable Diffusion XL (SDXL)** is an open-source text-to-image model by Stability AI. It generates high-resolution images (natively 1024×1024, MUSIA uses 1024×576 for 16:9 cinematic format).

### Core Components of SDXL in MUSIA:

#### 1. Dual Text Encoders
SDXL uses two text encoders (unlike older SD which used one):
- **CLIP ViT-L/14** → 768-dimensional embeddings
- **OpenCLIP ViT-bigG** → 1280-dimensional embeddings

Both outputs are concatenated → `prompt_embeds` shape = `(1, 77, 2048)`.

**This 2048 dimension is exactly why the PQC Enhancer uses `embed_dim=2048`** — it operates on the full concatenated SDXL text embedding.

#### 2. UNet (Denoiser)
The core of image generation:
- Takes noisy image latents + text embeddings as input
- Iteratively denoises the image over N steps (MUSIA uses **28 steps**)
- Uses **cross-attention** at each step — this is where text embeddings "guide" the image

```python
result = pipe(
    prompt_embeds=prompt_embeds,            # Quantum-enhanced (or standard) text embeddings
    negative_prompt_embeds=negative_embeds, # What to avoid
    num_inference_steps=28,                 # Denoising iterations
    guidance_scale=7.5,                     # CFG: how strongly to follow the prompt
    height=576, width=1024,                 # 16:9 cinematic aspect ratio
)
```

- `guidance_scale=7.5` — **Classifier-Free Guidance (CFG)**: the model generates both a "guided" image (following prompt) and an "unguided" image, then moves the output away from unguided and toward guided. 7.5 is a standard balanced value.

#### 3. VAE (Variational Autoencoder)
SDXL generates in **latent space** (8x compressed), then the VAE decoder converts latents to pixels. MUSIA enables memory optimizations:
- `enable_attention_slicing()` — splits attention computation into chunks → reduces VRAM peak
- `enable_vae_tiling()` / `enable_vae_slicing()` — handles large images without Out-of-Memory errors

#### 4. LoRA Adapter (Fine-Tuning for Cinematic Style)
**File:** `adapter_model.safetensors` (88.8 MB, 1120 tensors)

**LoRA (Low-Rank Adaptation)** fine-tunes SDXL for storyboard aesthetics:
- Instead of retraining all 2.6B parameters of SDXL's UNet...
- LoRA adds small **rank-32 matrices** alongside existing weight matrices
- These were trained on cinematic storyboard images, giving SDXL a specific visual style

```python
pipe.load_lora_weights("/content/drive/MyDrive/adapter_model.safetensors")
```

### SDXL + Quantum — How They Work Together:

```
Story Scene Text
    ↓
SDXL Dual Text Encoders → prompt_embeds (1, 77, 2048)
    ↓  [QUANTUM PATH ONLY]
PQC QuantumFeatureEnhancer:
    LayerNorm → Linear(2048→4) → Quantum Circuit → Linear(4→2048) → Residual Add
    → Enhanced prompt_embeds (1, 77, 2048) — quantum-modulated
    ↓
SDXL UNet (28 denoising steps, guidance_scale=7.5)
    - Cross-attention at every step reads quantum-enhanced embeddings
    ↓
VAE Decoder → 1024×576 PNG image
```

Quantum enhancement modifies the text embeddings **before** they enter the UNet's cross-attention. Every one of the 28 denoising steps reads these embeddings when deciding what to paint — so quantum modulation influences the entire image generation process.

### SDXL Memory Management in Colab:
```python
# After generation, aggressively free GPU memory to prevent OOM on next scene
del result, image, prompt_embeds, negative_prompt_embeds
gc.collect()
torch.cuda.empty_cache()
```
SDXL on T4 GPU (15GB VRAM) uses ~10GB peak. Without cleanup between scenes, VRAM fills up and crashes Colab.

---

## 9. Full Project Architecture

### Directory Structure Explained:

```
Main_musia_project/
├── frontend/
│   └── index.html              ← Single-page Quantum Studio web app (HTML/CSS/JS)
│
├── backend/
│   ├── __init__.py             ← Python package marker
│   ├── main.py                 ← FastAPI server: routes, middleware, pipeline controller
│   ├── story_parser.py         ← Multilingual story → scene segmentation engine
│   ├── generate_on_colab.py    ← Colab tunnel client + local fallback renderer
│   ├── quantum_pipeline.py     ← PQC model definition, weight loading, inference
│   ├── fix_emoji.py            ← Emoji normalization utility
│   └── static/
│       └── generated_scenes/   ← Saved PNG files for each scene
│
├── api/
│   └── index.py               ← Vercel serverless entry point
│
├── colab_gpu_server.py        ← Runs INSIDE Google Colab (Flask + SDXL + PQC + ngrok)
│
├── best_pqc_enhancer_fulldataset.pt  ← Trained PQC weights (94 KB)
├── adapter_model.safetensors         ← Trained SDXL LoRA adapter (88.8 MB)
├── requirements.txt            ← Python dependencies
├── vercel.json                 ← Vercel deployment configuration
└── tests/
    └── test_api.py             ← 19 async integration tests
```

### Complete End-to-End Flow Diagram:

```
┌─────────────────────────────────────────────────────────────┐
│                  USER'S BROWSER                              │
│                                                             │
│  MUSIA Quantum Studio (frontend/index.html)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. User pastes story text                          │   │
│  │  2. Clicks "Parse Story" → POST /api/parse-story    │   │
│  │  3. Scene cards rendered in timeline                │   │
│  │  4. Selects model (quantum / standard / mock)       │   │
│  │  5. Clicks "Launch Pipeline"                        │   │
│  │     → POST /api/start-generation                   │   │
│  │  6. Polls GET /api/generation-status every 3s       │   │
│  │  7. Images appear in gallery as they complete       │   │
│  └──────────────────────────┬──────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │ HTTP (port 8000)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (backend/main.py)               │
│                                                             │
│  Middleware: Rate limiting (120/min/IP), Security headers    │
│                                                             │
│  /api/parse-story                                           │
│    └→ story_parser.py                                       │
│         normalize() → detect_language() → split_sentences() │
│         → chunk(3 sentences/scene) → format_as_prompts()   │
│                                                             │
│  /api/start-generation                                      │
│    └→ run_generation_pipeline() [asyncio background task]   │
│         For each scene prompt:                              │
│           generate_on_colab.py → render_scene()            │
│                                                             │
│  /api/generation-status                                     │
│    └→ {is_running, current_scene, image_urls[]}            │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────────────────────┐
         │                                            │
         ▼ (Colab online + ngrok active)              ▼ (Colab offline / mock mode)
┌─────────────────────────┐          ┌──────────────────────────────────────┐
│   NGROK TUNNEL (HTTPS)  │          │  LOCAL FALLBACK (quantum_pipeline.py) │
│          ↓              │          │                                        │
│  GOOGLE COLAB GPU       │          │  compute_text_embedding(prompt)        │
│  (colab_gpu_server.py)  │          │  → tokenize Indic/Latin text           │
│                         │          │  → sinusoidal harmonics + lang_bias    │
│  Flask /generate:       │          │  → (1, seq_len, 2048) tensor           │
│  1. encode_prompt()     │          │         ↓                              │
│     [SDXL on CUDA GPU]  │          │  QuantumFeatureEnhancer                │
│  2. pqc_enhancer(embs)  │          │  → LayerNorm → Linear(2048→4)          │
│     [PQC on CPU]        │          │  → PennyLane Quantum Circuit           │
│  3. SDXL UNet 28 steps  │          │  → Linear(4→2048) → Residual add       │
│     [CUDA GPU]          │          │         ↓                              │
│  4. VAE decode → PNG    │          │  PIL Image (1024×576, 16:9)            │
│  5. Returns PNG bytes   │          │  → Gradient background (quantum-tinted)│
│                         │          │  → Letterbox bars + Director HUD       │
└─────────────────────────┘          │  → Language badge + scene text overlay │
         │                           │  → Quantum diagnostics: q0..q3 values  │
         │ PNG bytes / data URL      │  → Base64 data URL                     │
         └──────────────┬────────────└──────────────────────────────────────┘
                        ▼
           backend/static/generated_scenes/scene_N.png
                        ↓
           /api/generation-status returns image URL
                        ↓
           Frontend renders image in gallery in real-time
```

### Technology Stack Summary:

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML5 + Vanilla CSS + JavaScript | SPA Quantum Studio UI |
| Styling | Tailwind CSS (CDN) + Custom CSS Variables | Glassmorphism design |
| 3D Effects | Three.js | WebGL image generation shader |
| ZIP Export | JSZip | Download all scenes as ZIP archive |
| Backend Framework | FastAPI (Python) | REST API server |
| ASGI Server | Uvicorn | Serves FastAPI |
| Story Parsing | Custom Python (regex + unicodedata) | Multilingual scene segmentation |
| Quantum ML | PennyLane + PyTorch | 4-qubit PQC feature enhancer |
| Image Generation | SDXL (HuggingFace Diffusers) | Text-to-image diffusion |
| Fine-tuning | LoRA rank-32 adapter | Cinematic storyboard style |
| Cloud GPU | Google Colab (T4 / A100) | GPU inference server |
| Tunneling | pyngrok / ngrok | Expose Colab to internet |
| Colab HTTP Server | Flask | Lightweight GPU API (in Colab) |
| Deployment | Vercel (serverless) | Production frontend + API hosting |
| Testing | unittest + httpx ASGI | 19 async integration tests |
| Supported Languages | English, Hindi (Devanagari), Bengali (Bangla) | Multilingual support |

---

## 🔑 Key Concepts Summary

| Concept | Plain English Explanation |
|---|---|
| **PQC** (Parameterized Quantum Circuit) | A quantum circuit whose gate angles are learned from data — like a neural network but using quantum gates |
| **AngleEmbedding** | Encodes a classical number as a qubit rotation angle — maps data into quantum state space (Bloch sphere) |
| **BasicEntanglerLayers** | A quantum "layer" of parameterized rotations + CNOT gates that creates correlations between qubits |
| **Hilbert Space** | The mathematical space where quantum states live — exponentially richer than Euclidean space |
| **Expectation Value** | The average measurement result for a qubit — `+1` means likely `|0⟩`, `-1` means likely `|1⟩` |
| **SDXL** | Stable Diffusion XL — a 2.6B parameter open-source image generation model |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning by adding small extra matrices alongside large model weights |
| **CFG** (Classifier-Free Guidance) | How strongly the model follows the text prompt (7.5 = balanced quality vs creativity) |
| **VAE** | Variational Autoencoder — compresses images to latent space for diffusion, then reconstructs pixels |
| **ngrok** | Creates a secure public HTTPS URL that tunnels traffic into a private/Colab server |
| **Polling** | Frontend repeatedly asks backend "are you done yet?" instead of backend pushing (no WebSockets) |
| **NFC Normalization** | Unicode canonical form — ensures Indic script characters have consistent byte representations |
| **Purna Viram (।)** | The Hindi/Bengali full stop — sentence boundary marker equivalent to `.` in English |
| **LoRA rank-32** | Each LoRA matrix has inner dimension 32 — controls capacity vs. size tradeoff of the adapter |
| **asyncio background task** | FastAPI runs image generation in a background coroutine so the API stays responsive during generation |

---

*This guide was generated from a thorough analysis of all source files in the MUSIA project. Every section maps directly to actual code in the repository. Written for the purpose of complete project understanding — from first principles to production deployment.*
