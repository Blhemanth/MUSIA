"""
backend/render_demo_video.py
MUSIA - Multilingual Story Illustration
Comprehensive Video Generator: Programmatic 720p HD Cinematic Walkthrough Video
Renders both MP4 and animated WebP demonstrating the entire MUSIA project workflow.
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = r"C:\Users\heman\.gemini\antigravity-ide\brain\0ae1a8eb-e5c5-4070-8d72-91e60d9695c2"
SCENES_DIR = os.path.join(BASE_DIR, "backend", "static", "generated_scenes")

WIDTH = 1280
HEIGHT = 720
FPS = 12

# Fonts
FONT_BOLD_PATH = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REG_PATH = r"C:\Windows\Fonts\segoeui.ttf"
FONT_MONO_PATH = r"C:\Windows\Fonts\consola.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

font_h1 = get_font(FONT_BOLD_PATH, 38)
font_h2 = get_font(FONT_BOLD_PATH, 26)
font_h3 = get_font(FONT_BOLD_PATH, 20)
font_body = get_font(FONT_REG_PATH, 16)
font_sub = get_font(FONT_REG_PATH, 14)
font_small = get_font(FONT_REG_PATH, 12)
font_mono = get_font(FONT_MONO_PATH, 14)
font_mono_sm = get_font(FONT_MONO_PATH, 12)
font_mono_xs = get_font(FONT_MONO_PATH, 10)

# Colors (Cosmic / Quantum Palette)
C_BG = (4, 7, 20)
C_CARD = (10, 15, 36)
C_CARD_BORDER = (99, 102, 241)
C_CYAN = (0, 242, 254)
C_PURPLE = (121, 40, 202)
C_TEAL = (9, 211, 172)
C_INDIGO = (79, 70, 229)
C_TEXT_LIGHT = (241, 245, 249)
C_TEXT_MUTED = (148, 163, 184)
C_AMBER = (251, 191, 36)
C_RED = (239, 68, 68)

# Load real scene images if available
scene_imgs = {}
for i in [1, 2, 3, 4, 5, 6]:
    p = os.path.join(SCENES_DIR, f"scene_{i}.png")
    if os.path.exists(p):
        try:
            im = Image.open(p).convert("RGB")
            scene_imgs[i] = im
        except Exception as e:
            print(f"Warn: couldn't load scene_{i}: {e}")

# Pre-generate simulated quantum particles
np.random.seed(42)
PARTICLES = []
for _ in range(45):
    PARTICLES.append({
        'x': np.random.uniform(20, WIDTH - 20),
        'y': np.random.uniform(60, HEIGHT - 60),
        'vx': np.random.uniform(-0.8, 0.8),
        'vy': np.random.uniform(-0.8, 0.8),
        'size': np.random.uniform(1.5, 3.5),
        'color': C_CYAN if np.random.random() > 0.5 else (C_PURPLE if np.random.random() > 0.5 else C_INDIGO),
    })

def create_base_canvas(t_frame, act_index=0, act_title="", act_subtitle=""):
    """Creates base background with ambient glow, animated quantum particles, header & footer."""
    im = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(im)

    # 1. Ambient radial gradients
    # Top-left cyan glow
    glow_cyan = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow_cyan)
    g_draw.ellipse((-100, -100, 400, 400), fill=(0, 242, 254, 25))
    g_draw.ellipse((WIDTH - 300, HEIGHT - 300, WIDTH + 200, HEIGHT + 200), fill=(121, 40, 202, 35))
    im.paste(glow_cyan, (0, 0), glow_cyan)

    # 2. Animated quantum particles and entanglement threads
    for p in PARTICLES:
        px = (p['x'] + p['vx'] * t_frame) % (WIDTH - 40) + 20
        py = (p['y'] + p['vy'] * t_frame) % (HEIGHT - 120) + 60
        r = p['size']
        draw.ellipse((px - r, py - r, px + r, py + r), fill=p['color'])

    # Entanglement lines between nearby particles
    for i in range(0, len(PARTICLES), 2):
        p1 = PARTICLES[i]
        p2 = PARTICLES[(i + 1) % len(PARTICLES)]
        x1 = (p1['x'] + p1['vx'] * t_frame) % (WIDTH - 40) + 20
        y1 = (p1['y'] + p1['vy'] * t_frame) % (HEIGHT - 120) + 60
        x2 = (p2['x'] + p2['vx'] * t_frame) % (WIDTH - 40) + 20
        y2 = (p2['y'] + p2['vy'] * t_frame) % (HEIGHT - 120) + 60
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 120:
            draw.line((x1, y1, x2, y2), fill=(79, 70, 229, 60), width=1)

    # 3. Top Header Bar
    draw.rectangle((0, 0, WIDTH, 54), fill=(6, 11, 28))
    draw.line((0, 54, WIDTH, 54), fill=(99, 102, 241, 100), width=1)

    # Logo & Title
    draw.text((25, 14), "⚛  MUSIA", fill=C_CYAN, font=font_h2)
    draw.text((160, 19), "QUANTUM STUDIO v3.0", fill=C_TEXT_MUTED, font=font_mono_sm)

    # Engine Status Pill
    draw.rounded_rectangle((480, 12, 750, 42), radius=15, fill=(15, 23, 54), outline=(9, 211, 172, 180), width=1)
    draw.ellipse((495, 23, 503, 31), fill=C_TEAL)
    draw.text((512, 18), "ENGINE ACTIVE • CUDA • PENNYLANE PQC", fill=C_TEAL, font=font_mono_xs)

    # Right tech badges
    draw.rounded_rectangle((WIDTH - 380, 14, WIDTH - 260, 40), radius=6, fill=(18, 24, 50), outline=C_CARD_BORDER)
    draw.text((WIDTH - 370, 19), "SDXL LoRA Diff", fill=C_TEXT_LIGHT, font=font_mono_xs)

    draw.rounded_rectangle((WIDTH - 250, 14, WIDTH - 140, 40), radius=6, fill=(18, 24, 50), outline=C_CARD_BORDER)
    draw.text((WIDTH - 240, 19), "3 Languages", fill=C_TEXT_LIGHT, font=font_mono_xs)

    draw.rounded_rectangle((WIDTH - 130, 14, WIDTH - 25, 40), radius=6, fill=(18, 24, 50), outline=C_CARD_BORDER)
    draw.text((WIDTH - 120, 19), "FastAPI 3.0", fill=C_TEXT_LIGHT, font=font_mono_xs)

    # 4. Bottom Footer / Progress Bar
    draw.rectangle((0, HEIGHT - 50, WIDTH, HEIGHT), fill=(6, 11, 28))
    draw.line((0, HEIGHT - 50, WIDTH, HEIGHT - 50), fill=(99, 102, 241, 100), width=1)

    # 7-Step Navigation Tracker
    steps = ["1. INTRO", "2. ARCHITECTURE", "3. NLP PARSE", "4. QUANTUM PQC", "5. DIFFUSION", "6. GALLERY", "7. EXPORT"]
    step_w = (WIDTH - 50) // len(steps)
    for idx, s in enumerate(steps):
        sx = 25 + idx * step_w
        is_active = (idx == act_index)
        is_past = (idx < act_index)
        col = C_CYAN if is_active else (C_TEAL if is_past else C_TEXT_MUTED)
        bg_col = (15, 30, 65) if is_active else (10, 15, 30)
        draw.rounded_rectangle((sx, HEIGHT - 42, sx + step_w - 10, HEIGHT - 12), radius=5, fill=bg_col, outline=col if is_active else (40, 50, 80))
        draw.text((sx + 8, HEIGHT - 33), s, fill=col, font=font_mono_xs)

    return im, draw

def render_act_1_intro(frame_idx, total_frames):
    """Act 1: Title, Hero, and Overview Cards."""
    im, draw = create_base_canvas(frame_idx, act_index=0)
    
    # Hero Title Card
    card_box = (80, 85, WIDTH - 80, 310)
    draw.rounded_rectangle(card_box, radius=16, fill=(12, 18, 44), outline=C_CYAN, width=2)

    # Title Glow Effect
    draw.text((120, 110), "MUSIA — Multilingual Story Illustration", fill=C_CYAN, font=font_h1)
    draw.text((120, 165), "Cinematic Visual Storytelling Powered by Quantum-Enhanced Neural Diffusion", fill=C_TEXT_LIGHT, font=font_h2)
    
    desc_p1 = "Deconstructs written narratives into sequential, director-grade storyboards and generates high-fidelity"
    desc_p2 = "visual illustrations using a hybrid Parameterized Quantum Circuit (PQC) coupled with Stable Diffusion XL."
    draw.text((120, 215), desc_p1, fill=C_TEXT_MUTED, font=font_body)
    draw.text((120, 240), desc_p2, fill=C_TEXT_MUTED, font=font_body)

    # Highlights Row: 4 Feature Glass Cards
    cards = [
        ("🌐 Multilingual NLP", "English, Hindi, Bengali", "Purna Viram (।) boundary split"),
        ("⚛  4-Qubit PQC Circuit", "PennyLane AngleEmbedding", "16-dim Hilbert state modulation"),
        ("🎨 SDXL LoRA Diffusion", "Cloud GPU / Local Fallback", "High-resolution cinematic 16:9"),
        ("🎬 Director's Studio UI", "WebGL Shader Lightbox", "Interactive live timeline & export")
    ]
    card_w = (WIDTH - 160 - 45) // 4
    for i, (title, sub1, sub2) in enumerate(cards):
        cx = 80 + i * (card_w + 15)
        cy = 340
        cw = card_w
        ch = 260
        draw.rounded_rectangle((cx, cy, cx + cw, cy + ch), radius=12, fill=(14, 20, 48), outline=C_CARD_BORDER, width=1)
        
        # Header strip inside card
        draw.rounded_rectangle((cx + 12, cy + 15, cx + cw - 12, cy + 50), radius=6, fill=(24, 32, 70))
        draw.text((cx + 20, cy + 22), title, fill=C_CYAN, font=font_h3)

        draw.text((cx + 20, cy + 80), "• " + sub1, fill=C_TEXT_LIGHT, font=font_sub)
        draw.text((cx + 20, cy + 115), "• " + sub2, fill=C_TEXT_MUTED, font=font_sub)

        # Micro indicator at bottom of card
        draw.line((cx + 20, cy + ch - 40, cx + cw - 20, cy + ch - 40), fill=(40, 50, 90))
        draw.text((cx + 20, cy + ch - 30), "STATUS: VERIFIED", fill=C_TEAL, font=font_mono_xs)

    return im

def render_act_2_architecture(frame_idx, total_frames):
    """Act 2: Quantum Architecture & Hybrid Pipeline Flow."""
    im, draw = create_base_canvas(frame_idx, act_index=1)

    # Header Card
    draw.rounded_rectangle((80, 75, WIDTH - 80, 135), radius=12, fill=(12, 18, 44), outline=C_CARD_BORDER)
    draw.text((105, 88), "HYBRID QUANTUM-CLASSICAL PIPELINE ARCHITECTURE", fill=C_CYAN, font=font_h2)
    draw.text((105, 115), "PennyLane 4-Qubit PQC Hilbert Space Modulation + SDXL Latent Diffusion", fill=C_TEXT_MUTED, font=font_sub)

    # Flow Diagram: 5 Stage Blocks
    stages = [
        ("1. Narrative Script", "Prose in EN / HI / BN", C_INDIGO),
        ("2. Scene Chunking", "NLP sentence boundary", C_INDIGO),
        ("3. 4-Qubit PQC", "PennyLane Hilbert State", C_CYAN),
        ("4. SDXL LoRA", "Latent neural diffusion", C_PURPLE),
        ("5. Storyboard Cut", "16:9 cinematic render", C_TEAL),
    ]
    sw = (WIDTH - 160 - 60) // 5
    for i, (title, sub, col) in enumerate(stages):
        sx = 80 + i * (sw + 15)
        sy = 155
        draw.rounded_rectangle((sx, sy, sx + sw, sy + 75), radius=10, fill=(14, 22, 54), outline=col, width=2)
        draw.text((sx + 12, sy + 15), title, fill=col, font=font_h3)
        draw.text((sx + 12, sy + 45), sub, fill=C_TEXT_LIGHT, font=font_small)
        if i < 4:
            # Arrow
            draw.text((sx + sw + 3, sy + 25), "➔", fill=C_CYAN, font=font_h2)

    # Main Quantum Circuit Schematic Panel
    panel_box = (80, 250, WIDTH - 80, 640)
    draw.rounded_rectangle(panel_box, radius=14, fill=(10, 16, 40), outline=C_CYAN, width=2)

    draw.text((110, 268), "PENNYLANE PARAMETERIZED QUANTUM CIRCUIT (best_pqc_enhancer_fulldataset.pt)", fill=C_CYAN, font=font_h3)
    draw.text((110, 295), "Transforms 2048-dim CLIP embeddings through multi-qubit entangled rotations in complex Hilbert space:", fill=C_TEXT_MUTED, font=font_body)

    # Quantum Register Wires
    wire_names = ["q₀ (Register Wire 0)", "q₁ (Register Wire 1)", "q₂ (Register Wire 2)", "q₃ (Register Wire 3)"]
    pauli_vals = ["⟨Z₀⟩: +0.0518", "⟨Z₁⟩: -0.0237", "⟨Z₂⟩: -0.2738", "⟨Z₃⟩: -0.0138"]
    
    wire_y_start = 345
    wire_spacing = 65
    for idx, (wname, pval) in enumerate(zip(wire_names, pauli_vals)):
        wy = wire_y_start + idx * wire_spacing
        
        # Wire label
        draw.text((110, wy - 10), wname, fill=C_TEXT_LIGHT, font=font_mono_sm)

        # Base wire line
        draw.line((280, wy, WIDTH - 340, wy), fill=(80, 100, 160), width=2)

        # Gate 1: AngleEmbedding
        draw.rounded_rectangle((320, wy - 20, 440, wy + 20), radius=5, fill=(24, 38, 90), outline=C_CYAN)
        draw.text((330, wy - 8), "AngleEmbed(θ)", fill=C_CYAN, font=font_mono_xs)

        # Gate 2: Parameterized RY
        draw.rounded_rectangle((480, wy - 20, 560, wy + 20), radius=5, fill=(40, 25, 80), outline=C_PURPLE)
        draw.text((495, wy - 8), f"RY(w_{idx})", fill=C_PURPLE, font=font_mono_xs)

        # Entanglement CNOTs
        draw.ellipse((620, wy - 6, 632, wy + 6), fill=C_CYAN)
        if idx < 3:
            draw.line((626, wy, 626, wy + wire_spacing), fill=C_CYAN, width=2)
            draw.ellipse((621, wy + wire_spacing - 5, 631, wy + wire_spacing + 5), fill=C_PURPLE)

        # Measurement Gate & Pauli-Z HUD Box
        draw.rounded_rectangle((WIDTH - 320, wy - 20, WIDTH - 110, wy + 20), radius=6, fill=(15, 30, 60), outline=C_TEAL, width=2)
        draw.text((WIDTH - 305, wy - 9), pval, fill=C_TEAL, font=font_mono)

    # Bottom summary HUD
    draw.rounded_rectangle((110, 600, WIDTH - 110, 630), radius=6, fill=(15, 22, 55))
    draw.text((130, 607), "⚡ RESULT: Modulated prompt latent vectors preserve director sentiment & enhance cross-attention cohesion.", fill=C_AMBER, font=font_mono_xs)

    return im

def render_act_3_parsing(frame_idx, total_frames):
    """Act 3: Narrative Script Ingestion & Intelligent Story Parsing."""
    im, draw = create_base_canvas(frame_idx, act_index=2)

    # Split Screen: Left (Input Console) / Right (Storyboard Breakdown)
    mid_x = 520

    # Left: Prose Script Input
    draw.rounded_rectangle((60, 75, mid_x, 640), radius=14, fill=(12, 18, 44), outline=C_CARD_BORDER)
    draw.text((85, 95), "PROSE SCRIPT INPUT", fill=C_CYAN, font=font_h2)
    draw.text((85, 125), "Multilingual story narrative in English, Hindi, or Bengali", fill=C_TEXT_MUTED, font=font_small)

    # Narrative text area simulation
    story_lines = [
        "\"The Chrono-Observatory hovered silently above the",
        "rings of Saturn, its crystal dome reflecting bands of",
        "amber and ice.",
        "",
        "Inside the central chamber, Doctor Vane adjusted the",
        "quantum resonator. Golden particle streams coalesced",
        "into an ethereal doorway suspended in mid-air.",
        "",
        "Suddenly, a brilliant flare of turquoise energy erupted",
        "from the gateway, casting long neon shadows across the",
        "titanium floor as the first visitor stepped through.\""
    ]
    draw.rounded_rectangle((85, 155, mid_x - 25, 410), radius=10, fill=(6, 10, 26), outline=(60, 70, 110))
    for l_idx, line in enumerate(story_lines):
        draw.text((100, 170 + l_idx * 21), line, fill=C_TEXT_LIGHT, font=font_sub)

    # Character count & language detected banner
    draw.text((85, 425), "396 / 5,000 characters • 58 words", fill=C_TEXT_MUTED, font=font_mono_xs)

    draw.rounded_rectangle((85, 450, mid_x - 25, 495), radius=8, fill=(15, 35, 60), outline=C_CYAN)
    draw.ellipse((100, 468, 110, 478), fill=C_CYAN)
    draw.text((120, 463), "Detected: English (Latin Script) • Status: Validated", fill=C_CYAN, font=font_mono_sm)

    # Primary Action Button: "Analyze & Dissect Scenes"
    draw.rounded_rectangle((85, 520, mid_x - 25, 575), radius=10, fill=(79, 70, 229), outline=C_CYAN, width=2)
    draw.text((125, 535), "⚡ ANALYZE & DISSECT SCENES", fill=C_TEXT_LIGHT, font=font_h3)

    # Left Bottom Note
    draw.rounded_rectangle((85, 595, mid_x - 25, 625), radius=6, fill=(15, 20, 45))
    draw.text((95, 603), "NLP Engine: /api/parse-story-detailed (FastAPI)", fill=C_TEXT_MUTED, font=font_mono_xs)

    # Right: Storyboard Breakdown Cards
    draw.rounded_rectangle((mid_x + 25, 75, WIDTH - 60, 640), radius=14, fill=(10, 16, 40), outline=C_CYAN, width=2)
    draw.text((mid_x + 50, 95), "STORYBOARD BREAKDOWN (3 SCENES PARSED)", fill=C_CYAN, font=font_h2)
    draw.text((mid_x + 50, 125), "Chronologically segmented cinematic scene cues:", fill=C_TEXT_MUTED, font=font_small)

    scenes = [
        ("SCENE 01", "Chrono-Observatory hovering above Saturn's rings", "Crystal dome reflecting amber ice rings; space exterior wide.", "EST TIME: ~1.2s", C_CYAN),
        ("SCENE 02", "Doctor Vane & Quantum Resonator Activation", "Golden particle streams coalescing into an ethereal doorway.", "EST TIME: ~1.2s", C_PURPLE),
        ("SCENE 03", "Gateway Turquoise Flare & First Visitor", "Blinding turquoise energy flare, neon shadows on titanium floor.", "EST TIME: ~1.2s", C_TEAL),
    ]

    for s_idx, (sc_id, sc_title, sc_desc, sc_meta, sc_col) in enumerate(scenes):
        sc_y = 155 + s_idx * 155
        draw.rounded_rectangle((mid_x + 50, sc_y, WIDTH - 85, sc_y + 140), radius=10, fill=(15, 22, 55), outline=sc_col, width=2)
        
        # Scene badge
        draw.rounded_rectangle((mid_x + 65, sc_y + 15, mid_x + 165, sc_y + 45), radius=6, fill=(25, 35, 80), outline=sc_col)
        draw.text((mid_x + 75, sc_y + 22), sc_id, fill=sc_col, font=font_h3)

        draw.text((mid_x + 180, sc_y + 22), sc_title, fill=C_TEXT_LIGHT, font=font_h3)
        draw.text((mid_x + 65, sc_y + 60), sc_desc, fill=C_TEXT_MUTED, font=font_sub)

        # Meta tags
        draw.text((mid_x + 65, sc_y + 105), "PQC: ACTIVE", fill=C_TEAL, font=font_mono_xs)
        draw.text((mid_x + 180, sc_y + 105), "MODEL: SDXL LoRA", fill=C_CYAN, font=font_mono_xs)
        draw.text((mid_x + 360, sc_y + 105), sc_meta, fill=C_AMBER, font=font_mono_xs)

    return im

def render_act_4_diffusion(frame_idx, total_frames):
    """Act 4: Launching Diffusion Pipeline & Live Progress Bar."""
    im, draw = create_base_canvas(frame_idx, act_index=3)

    # Frame animation progress (0% -> 100%)
    progress_ratio = min(1.0, frame_idx / max(1, total_frames - 1))
    pct = int(progress_ratio * 100)
    current_scene_num = 1 if pct < 35 else (2 if pct < 75 else 3)

    # Console Card
    draw.rounded_rectangle((80, 75, WIDTH - 80, 640), radius=14, fill=(10, 16, 40), outline=C_CYAN, width=2)

    draw.text((110, 95), "DIFFUSION ENGINE EXECUTION CONSOLE", fill=C_CYAN, font=font_h2)
    draw.text((110, 125), "Synthesizing sequential cinematic frames via 4-Qubit PQC + SDXL Neural Pipeline", fill=C_TEXT_MUTED, font=font_sub)

    # Engine Status Info Box
    draw.rounded_rectangle((110, 160, WIDTH - 110, 240), radius=10, fill=(15, 24, 60), outline=C_CARD_BORDER)
    draw.text((135, 175), "TARGET ARCHITECTURE:", fill=C_TEXT_MUTED, font=font_mono_xs)
    draw.text((135, 195), "⚛  Quantum-Enhanced SDXL (Colab GPU / Local PQC Pipeline)", fill=C_CYAN, font=font_h3)

    draw.text((700, 175), "CURRENT STATUS:", fill=C_TEXT_MUTED, font=font_mono_xs)
    stat_msg = f"Synthesizing Scene {current_scene_num} of 3 ({pct}%)" if pct < 100 else "All 3 Scenes Successfully Generated!"
    draw.text((700, 195), stat_msg, fill=C_TEAL if pct == 100 else C_AMBER, font=font_h3)

    # Glowing Progress Bar
    bar_x = 110
    bar_y = 265
    bar_w = WIDTH - 220
    bar_h = 28
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=14, fill=(18, 24, 52), outline=(50, 60, 100))
    
    fill_w = int(bar_w * progress_ratio)
    if fill_w > 0:
        draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=14, fill=(0, 242, 254))

    draw.text((bar_x + bar_w - 75, bar_y - 25), f"{pct}%", fill=C_CYAN, font=font_mono)
    draw.text((bar_x, bar_y - 25), f"PROGRESS: Scene {current_scene_num} / 3", fill=C_TEXT_LIGHT, font=font_mono)

    # Active Live Scene Generation Preview Box
    box_y = 315
    draw.rounded_rectangle((110, box_y, WIDTH - 110, 615), radius=12, fill=(14, 20, 50), outline=C_CYAN if pct < 100 else C_TEAL, width=2)

    # Left: Active frame thumbnail
    th_w = 460
    th_h = 260
    draw.rounded_rectangle((135, box_y + 20, 135 + th_w, box_y + 20 + th_h), radius=10, fill=(6, 10, 26), outline=C_CARD_BORDER)

    # Draw real image if available
    img_to_show = scene_imgs.get(current_scene_num)
    if img_to_show:
        resized = img_to_show.resize((th_w - 4, th_h - 4))
        im.paste(resized, (137, box_y + 22))

    # Right: Real-time telemetry HUD
    hud_x = 135 + th_w + 35
    draw.text((hud_x, box_y + 25), f"SCENE {current_scene_num} QUANTUM DIAGNOSTICS", fill=C_CYAN, font=font_h3)

    telemetry = [
        ("PennyLane Register:", "4 Qubits (AngleEmbedding + BasicEntangler)"),
        ("Hilbert State Dimensions:", "16-Dimensional Complex Space (C¹⁶)"),
        ("Pauli-Z Expectations:", "[+0.0518, -0.0237, -0.2738, -0.0138]"),
        ("Quantum Latent Scale:", "1.000 (Residual Conditioning Active)"),
        ("LoRA Attention Rank:", "Rank 32 SDXL Adapter (88.8 MB)"),
        ("Inference Execution:", "CUDA Accelerated • PyTorch Bridge"),
        ("Frame Output Format:", "1024 x 576 Cinematic 16:9 PNG"),
    ]
    for t_idx, (k, v) in enumerate(telemetry):
        ty = box_y + 65 + t_idx * 30
        draw.text((hud_x, ty), k, fill=C_TEXT_MUTED, font=font_mono_sm)
        draw.text((hud_x + 220, ty), v, fill=C_TEXT_LIGHT, font=font_mono_sm)

    return im

def render_act_5_gallery(frame_idx, total_frames):
    """Act 5: Cinematic Storyboard Gallery Showcase."""
    im, draw = create_base_canvas(frame_idx, act_index=4)

    # Gallery Header Card
    draw.rounded_rectangle((60, 75, WIDTH - 60, 135), radius=12, fill=(12, 18, 44), outline=C_CYAN, width=2)
    draw.text((85, 88), "CINEMATIC SCENE GALLERY — DIRECTOR'S STORYBOARD", fill=C_CYAN, font=font_h2)
    draw.text((85, 115), "Sequential narrative frames synthesized with 4-Qubit PQC Hilbert Modulation", fill=C_TEXT_MUTED, font=font_sub)

    draw.rounded_rectangle((WIDTH - 280, 85, WIDTH - 80, 125), radius=8, fill=(15, 35, 70), outline=C_TEAL)
    draw.text((WIDTH - 265, 95), "3 FRAMES READY ✓", fill=C_TEAL, font=font_mono)

    # 3 Storyboard Frames Displayed Side by Side
    card_w = (WIDTH - 120 - 40) // 3
    card_h = 475
    y_top = 155

    captions = [
        ("SCENE 01: SATURN OBSERVATORY", "Chrono-Observatory hovering above amber rings of Saturn.", "q₀:+0.052  q₁:-0.024"),
        ("SCENE 02: QUANTUM RESONATOR", "Doctor Vane adjusting golden coalescing particle streams.", "q₂:-0.274  q₃:-0.014"),
        ("SCENE 03: GATEWAY ACTIVATION", "Brilliant turquoise flare erupting from the open ethereal portal.", "PQC Scale: 1.000"),
    ]

    for i in range(3):
        cx = 60 + i * (card_w + 20)
        draw.rounded_rectangle((cx, y_top, cx + card_w, y_top + card_h), radius=12, fill=(14, 20, 50), outline=C_CYAN, width=2)

        # Scene Frame Image Box
        img_h = 220
        draw.rounded_rectangle((cx + 10, y_top + 10, cx + card_w - 10, y_top + 10 + img_h), radius=8, fill=(6, 10, 26))

        sc_img = scene_imgs.get(i + 1)
        if sc_img:
            resized = sc_img.resize((card_w - 24, img_h - 4))
            im.paste(resized, (cx + 12, y_top + 12))

        # Scene metadata below image
        title, desc, qinfo = captions[i]
        draw.text((cx + 15, y_top + img_h + 25), title, fill=C_CYAN, font=font_h3)
        draw.text((cx + 15, y_top + img_h + 55), desc, fill=C_TEXT_LIGHT, font=font_small)

        # Quantum Badge Strip
        draw.rounded_rectangle((cx + 15, y_top + img_h + 115, cx + card_w - 15, y_top + img_h + 150), radius=6, fill=(18, 28, 65))
        draw.text((cx + 25, y_top + img_h + 125), "⚛  " + qinfo, fill=C_TEAL, font=font_mono_xs)

        # Interactive Button Simulations: "Cinema Lightbox" & "Quantum Compare"
        draw.rounded_rectangle((cx + 15, y_top + card_h - 60, cx + card_w // 2 - 5, y_top + card_h - 20), radius=6, fill=(24, 38, 90), outline=C_CYAN)
        draw.text((cx + 25, y_top + card_h - 47), "🔍 Lightbox", fill=C_CYAN, font=font_mono_xs)

        draw.rounded_rectangle((cx + card_w // 2 + 5, y_top + card_h - 60, cx + card_w - 15, y_top + card_h - 20), radius=6, fill=(40, 25, 80), outline=C_PURPLE)
        draw.text((cx + card_w // 2 + 15, y_top + card_h - 47), "⚛  Compare", fill=C_PURPLE, font=font_mono_xs)

    return im

def render_act_6_multilingual(frame_idx, total_frames):
    """Act 6: Multilingual Cross-Language Showcase (Hindi, Bengali, English)."""
    im, draw = create_base_canvas(frame_idx, act_index=5)

    # Header Card
    draw.rounded_rectangle((60, 75, WIDTH - 60, 135), radius=12, fill=(12, 18, 44), outline=C_CYAN, width=2)
    draw.text((85, 88), "MULTILINGUAL NARRATIVE SYNTHESIS ENGINE", fill=C_CYAN, font=font_h2)
    draw.text((85, 115), "Cross-platform support for Indic scripts (Hindi Devanagari, Bengali Bangla) and Latin English", fill=C_TEXT_MUTED, font=font_sub)

    # Two Large Language Cards Side by Side
    card_w = (WIDTH - 120 - 30) // 2
    card_h = 475
    y_top = 155

    # Card 1: Hindi (Devanagari)
    cx1 = 60
    draw.rounded_rectangle((cx1, y_top, cx1 + card_w, y_top + card_h), radius=14, fill=(14, 20, 50), outline=C_AMBER, width=2)
    draw.text((cx1 + 20, y_top + 20), "हिन्दी (HINDI DEVANAGARI)", fill=C_AMBER, font=font_h2)
    draw.text((cx1 + 20, y_top + 55), "Sample Narrative: Kashi Ganga Aarti Twilight", fill=C_TEXT_MUTED, font=font_sub)

    # Hindi script quote
    draw.rounded_rectangle((cx1 + 20, y_top + 85, cx1 + card_w - 20, y_top + 165), radius=8, fill=(8, 12, 30))
    draw.text((cx1 + 35, y_top + 98), "\"काशी के प्राचीन घाटों पर संध्या की आरती के दीप गंगा की", fill=C_TEXT_LIGHT, font=font_sub)
    draw.text((cx1 + 35, y_top + 123), "लहरों पर तैर रहे थे। नीले आकाश में तारों की छांव बिखर चुकी थी।\"", fill=C_TEXT_LIGHT, font=font_sub)

    # Illustration 1 thumbnail (Scene 4 / 2)
    img_h = 180
    sc_img1 = scene_imgs.get(4) or scene_imgs.get(2)
    if sc_img1:
        resized1 = sc_img1.resize((card_w - 44, img_h - 4))
        im.paste(resized1, (cx1 + 22, y_top + 185))

    # Hindi features
    draw.text((cx1 + 25, y_top + 390), "• Sentence Boundary: Purna Viram (।) automatic scene segmentation", fill=C_CYAN, font=font_small)
    draw.text((cx1 + 25, y_top + 415), "• CLIP Indic token conditioning with 4-qubit Pauli-Z expectations", fill=C_TEAL, font=font_small)
    draw.text((cx1 + 25, y_top + 440), "• Cohesive cinematic lighting & cultural motif preservation", fill=C_AMBER, font=font_small)

    # Card 2: Bengali (Bangla)
    cx2 = 60 + card_w + 30
    draw.rounded_rectangle((cx2, y_top, cx2 + card_w, y_top + card_h), radius=14, fill=(14, 20, 50), outline=C_PURPLE, width=2)
    draw.text((cx2 + 20, y_top + 20), "বাংলা (BENGALI BANGLA)", fill=C_PURPLE, font=font_h2)
    draw.text((cx2 + 20, y_top + 55), "Sample Narrative: Sundarbans Ancient Gateway", fill=C_TEXT_MUTED, font=font_sub)

    # Bengali script quote
    draw.rounded_rectangle((cx2 + 20, y_top + 85, cx2 + card_w - 20, y_top + 165), radius=8, fill=(8, 12, 30))
    draw.text((cx2 + 35, y_top + 98), "\"সুন্দরবনের গহীন অরণ্যে ঘন কুয়াশার চাদর ভেদ করে চাঁদের রূপালী", fill=C_TEXT_LIGHT, font=font_sub)
    draw.text((cx2 + 35, y_top + 123), "আলো নদীজলে ঝিকমিক করছিল। জলযানটি নিঃশব্দে এগিয়ে চলে।\"", fill=C_TEXT_LIGHT, font=font_sub)

    # Illustration 2 thumbnail (Scene 5 / 3)
    sc_img2 = scene_imgs.get(5) or scene_imgs.get(3)
    if sc_img2:
        resized2 = sc_img2.resize((card_w - 44, img_h - 4))
        im.paste(resized2, (cx2 + 22, y_top + 185))

    # Bengali features
    draw.text((cx2 + 25, y_top + 390), "• Script Normalization: Bangla Unicode (U+0980 to U+09FF)", fill=C_CYAN, font=font_small)
    draw.text((cx2 + 25, y_top + 415), "• Atmospheric fog & riverine lighting cross-attention alignment", fill=C_TEAL, font=font_small)
    draw.text((cx2 + 25, y_top + 440), "• Instant multilingual prompt normalization and validation", fill=C_PURPLE, font=font_small)

    return im

def render_act_7_export(frame_idx, total_frames):
    """Act 7: Master Storyboard Export & Outro."""
    im, draw = create_base_canvas(frame_idx, act_index=6)

    # Outro Card
    card_box = (100, 85, WIDTH - 100, 630)
    draw.rounded_rectangle(card_box, radius=16, fill=(12, 18, 44), outline=C_CYAN, width=2)

    draw.text((150, 120), "⚛  MUSIA — DIRECT YOUR STORY WITH QUANTUM PRECISION", fill=C_CYAN, font=font_h1)
    draw.text((150, 180), "Professional Multilingual Storyboard Generation is Complete & Verified", fill=C_TEXT_LIGHT, font=font_h2)

    # Export Menu Box
    draw.rounded_rectangle((150, 235, WIDTH - 150, 420), radius=12, fill=(16, 24, 60), outline=C_CARD_BORDER)
    draw.text((180, 255), "ONE-CLICK DIRECTOR'S MASTER EXPORT OPTIONS:", fill=C_AMBER, font=font_h3)

    exports = [
        ("📦 ZIP Package (PNGs)", "Full-resolution uncompressed 16:9 storyboards with scene cue naming"),
        ("📄 Scenes Data (JSON)", "Machine-readable script cues, token counts, and PennyLane Pauli-Z states"),
        ("🖨️ Printable Storyboard / PDF", "Clean printable director's proof sheet with scene annotations"),
    ]
    for e_idx, (etitle, edesc) in enumerate(exports):
        ey = 295 + e_idx * 38
        draw.text((180, ey), etitle, fill=C_CYAN, font=font_mono)
        draw.text((450, ey), edesc, fill=C_TEXT_MUTED, font=font_body)

    # Project Summary & Open Source Links
    draw.rounded_rectangle((150, 450, WIDTH - 150, 580), radius=12, fill=(8, 14, 35), outline=C_CYAN, width=1)
    draw.text((180, 470), "PROJECT ARCHITECTURE & REPOSITORIES", fill=C_CYAN, font=font_h3)
    draw.text((180, 505), "• Backend: FastAPI 3.0 (Async pipeline with Rate Limiting & Security)", fill=C_TEXT_LIGHT, font=font_sub)
    draw.text((180, 530), "• Quantum ML: PennyLane 4-Qubit PQC (AngleEmbedding + BasicEntangler)", fill=C_TEXT_LIGHT, font=font_sub)
    draw.text((680, 505), "• GitHub: github.com/Blhemanth/MUSIA", fill=C_TEAL, font=font_mono_sm)
    draw.text((680, 530), "• License: MIT Open Source • Built for Research & Cinema", fill=C_TEXT_MUTED, font=font_mono_sm)

    return im

def main():
    print("[VIDEO] Starting MUSIA Demo Video Generator...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    mp4_out_path = os.path.join(BASE_DIR, "musia_project_demo.mp4")
    webp_out_path = os.path.join(BASE_DIR, "musia_project_demo.webp")
    artifact_mp4_path = os.path.join(ARTIFACT_DIR, "musia_project_demo.mp4")
    artifact_webp_path = os.path.join(ARTIFACT_DIR, "musia_project_demo.webp")

    # Acts configuration (act_func, frame_count)
    acts = [
        (render_act_1_intro, 36),        # 3.0s
        (render_act_2_architecture, 48), # 4.0s
        (render_act_3_parsing, 54),      # 4.5s
        (render_act_4_diffusion, 60),    # 5.0s
        (render_act_5_gallery, 60),      # 5.0s
        (render_act_6_multilingual, 50), # 4.2s
        (render_act_7_export, 40),       # 3.3s
    ]

    total_frames_all = sum(count for _, count in acts)
    print(f"[VIDEO] Total frames to generate: {total_frames_all} at {FPS} FPS (~{total_frames_all/FPS:.1f} seconds)")

    # Initialize OpenCV VideoWriter for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(mp4_out_path, fourcc, FPS, (WIDTH, HEIGHT))

    frames_pil = []
    global_frame = 0

    for act_idx, (act_fn, num_frames) in enumerate(acts):
        print(f"[VIDEO] Rendering Act {act_idx + 1}/{len(acts)} ({num_frames} frames)...")
        for f in range(num_frames):
            im = act_fn(f, num_frames)
            
            # Write to MP4 (convert RGB -> BGR for OpenCV)
            np_frame = np.array(im)
            bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

            # Store for WebP animation (subsample every 2nd frame if needed to keep WebP ultra-crisp & lightweight)
            if global_frame % 2 == 0:
                # Resize slightly for WebP to keep file size under 15MB
                webp_im = im.resize((960, 540), Image.Resampling.LANCZOS)
                frames_pil.append(webp_im)

            global_frame += 1

    video_writer.release()
    print(f"[OK] MP4 Video generated successfully -> {mp4_out_path} ({os.path.getsize(mp4_out_path):,} bytes)")

    # Save animated WebP
    print("[VIDEO] Encoding animated WebP...")
    if frames_pil:
        frame_duration_ms = int(1000 / (FPS / 2)) # ~166ms
        frames_pil[0].save(
            webp_out_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=frame_duration_ms,
            loop=0,
            quality=85,
            method=4,
        )
        print(f"[OK] WebP Video generated successfully -> {webp_out_path} ({os.path.getsize(webp_out_path):,} bytes)")

    # Copy both files to artifact directory
    import shutil
    shutil.copy2(mp4_out_path, artifact_mp4_path)
    shutil.copy2(webp_out_path, artifact_webp_path)
    print(f"[OK] Copied videos to Artifact directory: {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
