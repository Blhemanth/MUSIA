"""
backend/story_parser.py
MUSIA - Multilingual Story Illustration
Phase 1 & 2: Multilingual Story-to-Scene Parser & Tokenizer

Natively supports:
1. English (Latin script)
2. Hindi (हिन्दी / Devanagari script: U+0900 - U+097F)
3. Bengali (বাংলা script: U+0980 - U+09FF)

Features:
- Unicode NFC normalization & Indic ZWJ/ZWNJ preservation
- Script and language identification
- Native sentence boundary detection supporting Western punctuation (. ! ?)
  and Indic sentence terminators:
    • Purna Viram '।' (U+0964)
    • Double Purna Viram '॥' (U+0965)
- Multilingual abbreviation protection
- Multilingual chapter and scene header recognition (English, Hindi, Bengali)
- Intelligent paragraph and sentence chunking into discrete sequential scenes (Scene 1, Scene 2, etc.)
- Script-aware multilingual tokenization
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Language & Script Detection
# ---------------------------------------------------------------------------

DEVANAGARI_RANGE = (0x0900, 0x097F)
BENGALI_RANGE    = (0x0980, 0x09FF)


def detect_language(text: str) -> str:
    """
    Detect the primary language/script of the input text.
    Returns: 'hi' (Hindi), 'bn' (Bengali), 'en' (English), or 'mixed'.
    """
    counts = {"hi": 0, "bn": 0, "en": 0}
    for char in text:
        cp = ord(char)
        if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
            counts["hi"] += 1
        elif BENGALI_RANGE[0] <= cp <= BENGALI_RANGE[1]:
            counts["bn"] += 1
        elif (65 <= cp <= 90) or (97 <= cp <= 122):
            counts["en"] += 1

    total = sum(counts.values())
    if total == 0:
        return "en"

    primary_lang, max_count = max(counts.items(), key=lambda item: item[1])
    if max_count / total >= 0.60:
        return primary_lang
    return "mixed"


# ---------------------------------------------------------------------------
# Multilingual Abbreviations & Sentence Splitting
# ---------------------------------------------------------------------------

MULTILINGUAL_ABBREVIATIONS = [
    # English
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.",
    "e.g.", "i.e.", "St.", "Lt.", "Capt.", "Gen.", "Col.", "No.", "Fig.",
    # Hindi (Devanagari)
    "पं.", "डॉ.", "डाॅ.", "प्रो.", "श्री.", "श्रीमती.", "कु.", "उदा.",
    "आदि.", "सं.", "ले.", "क्र.", "पृ.",
    # Bengali
    "ড.", "প্রা.", "শ্রী.", "শ্রীমতি.", "মু.", "মো.", "উদা.",
    "ইত্যাদি.", "নং.", "পৃ."
]

# Multilingual Chapter & Section Header Patterns
_CHAPTER_PATTERN = re.compile(
    r"^\s*(?:(?:chapter|part|section|scene|act|prologue|epilogue)\b[^\n:]*|"
    r"(?:अध्याय|भाग|दृश्य|खंड|अंक|प्रस्तावना|उपसंहार)[^\n:]*|"
    r"(?:অধ্যায়|পর্ব|বিভাগ|দৃশ্য|অঙ্ক|ভূমিকা|উপসংহার)[^\n:]*)",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_multilingual_text(text: str) -> str:
    """
    Applies Unicode NFC normalization to guarantee consistent canonical
    representations for Hindi and Bengali conjuncts while preserving zero-width
    characters (ZWJ, ZWNJ) required for Indic scripts.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def split_sentences_multilingual(text: str) -> List[str]:
    """
    Robust sentence splitter natively handling:
    - English (Latin) with '.', '!', '?'
    - Hindi (Devanagari) with Purna Viram '।', Double Purna Viram '॥', '?', '!', '.'
    - Bengali with Dari / Purna Viram '।', Double Purna Viram '॥', '?', '!', '.'

    Protects abbreviations, numbers with decimals (e.g. 3.14, ३.१४, ৩.১৪),
    and correctly handles scripts that do not have capital letter boundaries.
    """
    if not text or not text.strip():
        return []

    clean_text = normalize_multilingual_text(text)

    placeholder_map: Dict[str, str] = {}
    protected = clean_text

    # Protect abbreviations
    for i, abbr in enumerate(MULTILINGUAL_ABBREVIATIONS):
        token = f"__MUSIA_ABBR_{i}__"
        if abbr in protected:
            placeholder_map[token] = abbr
            protected = protected.replace(abbr, token)

    # Protect numbers with decimals
    def _mask_decimal(match):
        num_token = f"__MUSIA_DEC_{len(placeholder_map)}__"
        placeholder_map[num_token] = match.group(0)
        return num_token

    protected = re.sub(r'(\d|[०-९]|[০-৯])\s*\.\s*(\d|[०-९]|[০-৯])', _mask_decimal, protected)

    # Split on sentence terminators followed by whitespace or line breaks
    split_pattern = r'(?:(?<=[।॥!?])\s+)|(?:(?<=\.)\s+)'
    raw_sentences = re.split(split_pattern, protected)

    sentences: List[str] = []
    for s in raw_sentences:
        for token, original in placeholder_map.items():
            if token in s:
                s = s.replace(token, original)
        s = s.strip()
        if s and len(s) > 1:
            sentences.append(s)

    # Fallback if text didn't split but has purna viram
    if len(sentences) <= 1 and any(pv in clean_text for pv in ("।", "॥")):
        alt_splits = re.split(r'[।॥]+', clean_text)
        alt_sentences = [p.strip() for p in alt_splits if p.strip()]
        if len(alt_sentences) > len(sentences):
            sentences = alt_sentences

    return sentences


def tokenize_multilingual(text: str) -> List[str]:
    """
    Multilingual word/subword tokenizer for English, Hindi, and Bengali.
    Preserves Indic character sequences, matras, and diacritics.
    """
    clean_text = normalize_multilingual_text(text)
    token_pattern = re.compile(
        r'[\u0900-\u097F\u200C\u200D]+|'  # Devanagari words including ZWJ/ZWNJ
        r'[\u0980-\u09FF\u200C\u200D]+|'  # Bengali words including ZWJ/ZWNJ
        r'[A-Za-z0-9_\'-]+|'               # Latin words and digits
        r'[^\s\w]'                         # Punctuation marks
    )
    return token_pattern.findall(clean_text)


# ---------------------------------------------------------------------------
# Narrative Scene Segmentation
# ---------------------------------------------------------------------------

def _has_chapter_markers(text: str) -> bool:
    return bool(_CHAPTER_PATTERN.search(text))


def _split_by_chapters(text: str) -> List[str]:
    """Split story on chapter / part / section headers across all supported languages."""
    matches = list(_CHAPTER_PATTERN.finditer(text))
    if not matches:
        return [text]

    chunks: List[str] = []
    # If text exists before the first chapter header
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre and len(pre.split()) > 2:
            chunks.append(pre)

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(0).strip()
        body = text[start:end].strip()
        body = re.sub(r"^[:\s\-\–\—]+", "", body).strip()
        if body:
            full_segment = f"{header}: {body}" if header else body
            chunks.append(full_segment)
        elif header:
            chunks.append(header)

    return chunks


def _split_by_paragraphs(text: str) -> List[str]:
    """Split on double newlines / blank lines."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _chunk_sentences(sentences: List[str], chunk_size: int = 3) -> List[str]:
    """Group sentences into discrete scenes of `chunk_size`."""
    chunks: List[str] = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i: i + chunk_size]
        chunks.append(" ".join(group))
    return chunks


# ---------------------------------------------------------------------------
# Public Story Parser API
# ---------------------------------------------------------------------------

def parse_story_to_scenes(full_story: str, sentences_per_scene: int = 3) -> List[str]:
    """
    Parse raw story text in English, Hindi, or Bengali into sequential
    cinematic scene prompt strings.

    Output format:
        "Cinematic story illustration, scene {idx}: {scene_text}"

    Args:
        full_story: The raw narrative string.
        sentences_per_scene: Target sentences per scene for chunking.

    Returns:
        List of formatted cinematic scene prompts.
    """
    if not full_story or not full_story.strip():
        return []

    text = normalize_multilingual_text(full_story.strip())

    # Strategy 1: Chapter / Part markers (English, Hindi, Bengali)
    if _has_chapter_markers(text):
        raw_chunks = _split_by_chapters(text)
        final_chunks: List[str] = []
        for chapter_body in raw_chunks:
            sentences = split_sentences_multilingual(chapter_body)
            if len(sentences) <= sentences_per_scene:
                final_chunks.append(chapter_body.strip())
            else:
                final_chunks.extend(_chunk_sentences(sentences, sentences_per_scene))

    # Strategy 2: Paragraph breaks
    elif len(_split_by_paragraphs(text)) >= 2:
        paragraphs = _split_by_paragraphs(text)
        final_chunks = []
        for para in paragraphs:
            sentences = split_sentences_multilingual(para)
            if len(sentences) <= sentences_per_scene:
                final_chunks.append(para)
            else:
                final_chunks.extend(_chunk_sentences(sentences, sentences_per_scene))

    # Strategy 3: Multilingual sentence chunking
    else:
        sentences = split_sentences_multilingual(text)
        if len(sentences) <= sentences_per_scene:
            final_chunks = [text]
        else:
            final_chunks = _chunk_sentences(sentences, sentences_per_scene)

    # Format into sequential cinematic scene prompts
    scenes: List[str] = []
    for idx, chunk in enumerate(final_chunks, start=1):
        chunk = chunk.strip()
        if chunk:
            prompt = f"Cinematic story illustration, scene {idx}: {chunk}"
            scenes.append(prompt)

    return scenes


def parse_story_scenes_detailed(full_story: str, sentences_per_scene: int = 3) -> List[Dict[str, Any]]:
    """
    Detailed scene analysis returning metadata for each segmented scene,
    including detected language, script, token counts, and formatted prompt.
    """
    scene_prompts = parse_story_to_scenes(full_story, sentences_per_scene)
    results = []

    for idx, prompt in enumerate(scene_prompts, start=1):
        raw_scene_text = re.sub(r"^Cinematic story illustration, scene \d+:\s*", "", prompt, flags=re.IGNORECASE)
        lang = detect_language(raw_scene_text)
        tokens = tokenize_multilingual(raw_scene_text)

        results.append({
            "scene_id": idx,
            "scene_title": f"Scene {idx}",
            "language": lang,
            "language_name": {"en": "English", "hi": "Hindi (Devanagari)", "bn": "Bengali (Bangla)", "mixed": "Multilingual"}.get(lang, "English"),
            "raw_text": raw_scene_text,
            "prompt": prompt,
            "token_count": len(tokens),
            "tokens": tokens[:20],
        })

    return results
