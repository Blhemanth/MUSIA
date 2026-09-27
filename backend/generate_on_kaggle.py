import os
import requests

# Your active Google Colab ngrok tunnel URL
COLAB_NGROK_URL = "https://lend-eggplant-majesty.ngrok-free.dev"

def generate_scene_image(scene_prompt: str, scene_id: int, use_quantum: bool = True) -> str:
    """
    Sends the scene prompt and generation parameters to the live Google Colab GPU server,
    receives the generated image, and saves it locally for the FastAPI static file server.
    """
    url = f"{COLAB_NGROK_URL}/generate"
    payload = {
        "prompt": scene_prompt,
        "use_quantum": use_quantum,
        "scene_id": scene_id
    }
    
    print(f"🚀 Sending request to Google Colab GPU for Scene {scene_id} (Quantum: {use_quantum})...")
    
    try:
        # Request with a 300-second timeout to give the diffusion model time to generate
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code == 200:
            os.makedirs("backend/static/generated_scenes", exist_ok=True)
            local_path = f"backend/static/generated_scenes/scene_{scene_id}.png"
            
            with open(local_path, "wb") as f:
                f.write(response.content)
                
            print(f"✅ Scene {scene_id} image received from Colab and saved successfully!")
            return f"/static/generated_scenes/scene_{scene_id}.png"
        else:
            raise Exception(f"Colab Server Error ({response.status_code}): {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to Google Colab tunnel: {str(e)}")


# ---------------------------------------------------------------------------
# Mock / Fast-Preview Generator
# ---------------------------------------------------------------------------

OUTPUT_DIR = "backend/static/generated_scenes"

def generate_mock_scene_image(scene_prompt: str, scene_id: int) -> str:
    """
    Instantly generate a placeholder scene illustration (no GPU / Colab
    needed).  Creates a 512×512 gradient image with a text overlay of the
    scene prompt using Pillow so the frontend can be tested end-to-end in
    under a second.

    Args:
        scene_prompt: Cinematic prompt string for this scene.
        scene_id:     1-based scene index (used for the output filename).

    Returns:
        Relative HTTP URL: ``/static/generated_scenes/scene_<scene_id>.png``
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    import hashlib

    W, H = 512, 512

    # Derive deterministic palette from the prompt so each scene looks unique
    prompt_hash = hashlib.md5(scene_prompt.encode("utf-8")).hexdigest()
    r1 = int(prompt_hash[0:2], 16)
    g1 = int(prompt_hash[2:4], 16)
    b1 = int(prompt_hash[4:6], 16)
    r2 = int(prompt_hash[6:8], 16)
    g2 = int(prompt_hash[8:10], 16)
    b2 = int(prompt_hash[10:12], 16)

    # Build a vertical gradient
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Semi-transparent overlay band for text readability
    overlay = Image.new("RGBA", (W, 180), (0, 0, 0, 140))
    img = img.convert("RGBA")
    img.paste(overlay, (0, H - 180), overlay)

    draw = ImageDraw.Draw(img)

    # Pick a font (fall back to default bitmap font if no TTF available)
    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_body  = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font_title = ImageFont.load_default()
        font_body  = font_title

    # Title
    draw.text((20, H - 170), f"Scene {scene_id}  ·  Mock Preview", fill=(255, 255, 255, 230), font=font_title)

    # Wrapped prompt excerpt
    short_prompt = scene_prompt[:200]
    lines = textwrap.wrap(short_prompt, width=55)
    y_text = H - 140
    for line in lines[:5]:
        draw.text((20, y_text), line, fill=(210, 215, 230, 200), font=font_body)
        y_text += 18

    # Subtle "MOCK" watermark
    draw.text((W - 80, 16), "MOCK", fill=(255, 255, 255, 50), font=font_title)

    img = img.convert("RGB")

    # Save to the standard output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dest_filename = f"scene_{scene_id}.png"
    dest_path = os.path.join(OUTPUT_DIR, dest_filename)
    img.save(dest_path)

    relative_url = f"/static/generated_scenes/{dest_filename}"
    print(f"🖼️  [MOCK] Scene {scene_id} placeholder saved → {dest_path}")
    return relative_url