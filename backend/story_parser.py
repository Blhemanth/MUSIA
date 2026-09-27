"""
backend/story_parser.py
MUSIA - Multilingual Story Illustration
Phase 1: Story-to-Scene Parser

Splits raw story text into cinematic scene prompts using intelligent
paragraph-break detection, chapter-header recognition, and sentence chunking.
"""

import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Naïve but robust sentence splitter that handles common abbreviations.
    Returns a list of non-empty sentence strings.
    """
    # Protect common abbreviations so they don't trigger a split
    abbrevs = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
                "vs.", "etc.", "e.g.", "i.e.", "St.", "Lt.", "Capt."]
    placeholder_map: dict[str, str] = {}
    protected = text
    for i, abbr in enumerate(abbrevs):
        token = f"__ABBR{i}__"
        placeholder_map[token] = abbr
        protected = protected.replace(abbr, token)

    # Split on sentence-ending punctuation followed by whitespace + capital letter
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u2018\u2019\u201c\u201d])', protected)

    sentences = []
    for s in raw_sentences:
        # Restore abbreviations
        for token, abbr in placeholder_map.items():
            s = s.replace(token, abbr)
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


_CHAPTER_PATTERN = re.compile(
    r"^\s*(chapter\s+[\divxlc]+|part\s+[\divxlc]+|section\s+[\divxlc]+|prologue|epilogue|act\s+[\divxlc]+)",
    re.IGNORECASE | re.MULTILINE,
)


def _has_chapter_markers(text: str) -> bool:
    return bool(_CHAPTER_PATTERN.search(text))


def _split_by_chapters(text: str) -> list[str]:
    """Split story on chapter / part / section headers."""
    parts = _CHAPTER_PATTERN.split(text)
    # parts[0] is text before first header; subsequent pairs are (header, body)
    chunks: list[str] = []
    i = 0
    while i < len(parts):
        segment = parts[i].strip()
        if segment:
            chunks.append(segment)
        i += 1
    return [c for c in chunks if len(c.split()) > 5]  # drop trivially short fragments


def _split_by_paragraphs(text: str) -> list[str]:
    """Split on blank lines (double newlines)."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _chunk_sentences(sentences: list[str], chunk_size: int = 3) -> list[str]:
    """Group sentences into chunks of `chunk_size`."""
    chunks: list[str] = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i: i + chunk_size]
        chunks.append(" ".join(group))
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_story_to_scenes(full_story: str, sentences_per_scene: int = 3) -> list[str]:
    """
    Parse raw story text into a list of cinematic scene prompt strings.

    Strategy (in order of preference):
    1. If chapter / part / section markers exist → split by chapters.
    2. Else if multiple paragraphs exist (≥ 2 non-empty paragraphs) → split by paragraphs.
    3. Else → sentence-chunk the entire text (~`sentences_per_scene` per scene).

    Each chunk is then wrapped into a cinematic prompt:
        "Cinematic story illustration, scene X: <chunk text>"

    Args:
        full_story:          The raw story string.
        sentences_per_scene: Sentences per scene when falling back to chunking.

    Returns:
        A list of cinematic scene prompt strings.
    """
    if not full_story or not full_story.strip():
        return []

    text = full_story.strip()

    # --- Strategy 1: Chapter markers ---
    if _has_chapter_markers(text):
        raw_chunks = _split_by_chapters(text)
        # Each chapter itself may be long — further chunk by sentences
        final_chunks: list[str] = []
        for chapter_body in raw_chunks:
            sentences = _split_sentences(chapter_body)
            if len(sentences) <= sentences_per_scene:
                final_chunks.append(chapter_body.strip())
            else:
                final_chunks.extend(_chunk_sentences(sentences, sentences_per_scene))

    # --- Strategy 2: Paragraph breaks ---
    elif len(_split_by_paragraphs(text)) >= 2:
        paragraphs = _split_by_paragraphs(text)
        final_chunks = []
        for para in paragraphs:
            sentences = _split_sentences(para)
            if len(sentences) <= sentences_per_scene:
                final_chunks.append(para)
            else:
                final_chunks.extend(_chunk_sentences(sentences, sentences_per_scene))

    # --- Strategy 3: Sentence chunking ---
    else:
        sentences = _split_sentences(text)
        final_chunks = _chunk_sentences(sentences, sentences_per_scene)

    # Build cinematic prompts
    scenes: list[str] = []
    for idx, chunk in enumerate(final_chunks, start=1):
        chunk = chunk.strip()
        if chunk:
            prompt = f"Cinematic story illustration, scene {idx}: {chunk}"
            scenes.append(prompt)

    return scenes
