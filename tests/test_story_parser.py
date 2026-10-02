"""
tests/test_story_parser.py
Unit tests for the multilingual narrative dissection engine.
Supports English, Hindi (Devanagari), and Bengali (Bangla).
"""

import unittest
from backend.story_parser import (
    parse_story_to_scenes,
    parse_story_scenes_detailed,
    detect_language,
    normalize_multilingual_text,
    tokenize_multilingual,
)


class TestStoryParser(unittest.TestCase):
    """Test suite for story parsing and multilingual NLP utilities."""

    def setUp(self):
        self.sample_en = (
            "Chapter 1: The Beginning\n\n"
            "In a neon-lit cyberpunk metropolis, Maya looked across the sprawling skyline. "
            "Hovering vehicles weaved between monolithic skyscrapers under continuous rain.\n\n"
            "Chapter 2: The Encounter\n\n"
            "Deep in an underground data vault, she uncovered an ancient quantum cryptographic core. "
            "A holographic apparition materialized, warning her of the approaching security sentinels."
        )

        self.sample_hi = (
            "अध्याय १: यात्रा की शुरुआत\n\n"
            "एक प्राचीन नगर में एक साहसी योद्धा रहता था। "
            "उसने अपनी तलवार उठाई और घने जंगल की ओर चल पड़ा।\n\n"
            "अध्याय २: रहस्यमयी गुफा\n\n"
            "जंगल के बीच उसे एक चमकती हुई गुफा मिली। "
            "गुफा के अंदर एक दिव्य प्रकाश जल रहा था।"
        )

        self.sample_bn = (
            "অধ্যায় ১: নদীর তীরে\n\n"
            "গোধূলির আলোয় শান্ত নদীর বুক চিরে একটি নৌকা ভেসে যাচ্ছিল। "
            "আকাশে মেঘের রঙ রক্তিম রূপ ধারণ করেছিল।\n\n"
            "অধ্যায় ২: প্রাচীন মন্দির\n\n"
            "নদীর বাঁকে একটি শতাব্দীর প্রাচীন মন্দিরের চূড়া দৃশ্যমান হলো।"
        )

    def test_detect_language_english(self):
        self.assertEqual(detect_language("The quick brown fox jumps over the lazy dog."), "en")

    def test_detect_language_hindi(self):
        self.assertEqual(detect_language("नमस्ते, यह एक हिंदी कहानी का परीक्षण है।"), "hi")

    def test_detect_language_bengali(self):
        self.assertEqual(detect_language("আজকের দিনটি অপূর্ব সুন্দর এবং শান্ত।"), "bn")

    def test_normalize_multilingual_text(self):
        raw = "Hello world\r\nthis is\ra test"
        normalized = normalize_multilingual_text(raw)
        self.assertNotIn("\r", normalized)
        self.assertEqual(normalized, "Hello world\nthis is\na test")

    def test_tokenize_multilingual(self):
        tokens_en = tokenize_multilingual("Cyberpunk city in rain")
        self.assertGreater(len(tokens_en), 0)

        tokens_hi = tokenize_multilingual("एक साहसी योद्धा")
        self.assertGreater(len(tokens_hi), 0)

    def test_parse_story_to_scenes_english(self):
        scenes = parse_story_to_scenes(self.sample_en)
        self.assertIsInstance(scenes, list)
        self.assertGreaterEqual(len(scenes), 2)
        for scene in scenes:
            self.assertIsInstance(scene, str)
            self.assertGreater(len(scene.strip()), 0)

    def test_parse_story_to_scenes_hindi(self):
        scenes = parse_story_to_scenes(self.sample_hi)
        self.assertIsInstance(scenes, list)
        self.assertGreaterEqual(len(scenes), 2)
        # Should preserve Indic characters
        combined = " ".join(scenes)
        self.assertTrue(any('\u0900' <= char <= '\u097f' for char in combined))

    def test_parse_story_to_scenes_bengali(self):
        scenes = parse_story_to_scenes(self.sample_bn)
        self.assertIsInstance(scenes, list)
        self.assertGreaterEqual(len(scenes), 2)
        combined = " ".join(scenes)
        self.assertTrue(any('\u0980' <= char <= '\u09ff' for char in combined))

    def test_parse_story_scenes_detailed_structure(self):
        details = parse_story_scenes_detailed(self.sample_en)
        self.assertIsInstance(details, list)
        self.assertGreater(len(details), 0)

        first_scene = details[0]
        self.assertIn("scene_id", first_scene)
        self.assertIn("scene_title", first_scene)
        self.assertIn("prompt", first_scene)
        self.assertIn("raw_text", first_scene)
        self.assertIn("language", first_scene)
        self.assertIn("token_count", first_scene)
        self.assertEqual(first_scene["scene_id"], 1)

    def test_empty_story_handling(self):
        scenes = parse_story_to_scenes("")
        self.assertEqual(scenes, [])

        scenes_whitespace = parse_story_to_scenes("   \n\t  ")
        self.assertEqual(scenes_whitespace, [])


if __name__ == "__main__":
    unittest.main()
