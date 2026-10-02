"""Fix all emoji in print statements across backend Python files to ensure
Windows cp1252 compatibility when running under uvicorn."""

EMOJI_MAP = {
    "🎬": "[SCENE]",
    "🚀": "[START]",
    "⛔": "[STOP]",
    "✅": "[OK]",
    "⚠️": "[WARN]",
    "🖼️": "[IMG]",
    "❌": "[ERR]",
    "🏁": "[DONE]",
    "⚛️": "[MUSIA]",
    "ℹ️": "[INFO]",
}

FILES = [
    "backend/generate_on_colab.py",
    "backend/main.py",
    "backend/story_parser.py",
    "backend/quantum_pipeline.py",
]

for fname in FILES:
    with open(fname, encoding="utf-8") as f:
        data = f.read()

    changed = False
    for emoji, label in EMOJI_MAP.items():
        if emoji in data:
            data = data.replace(emoji, label)
            changed = True

    if changed:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"[FIXED] {fname}")
    else:
        print(f"[CLEAN] {fname}")

print("All backend files cleaned of emoji.")
