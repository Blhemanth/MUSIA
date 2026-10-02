"""
tests/test_api.py
Integration tests for the FastAPI REST endpoints using httpx ASGITransport
and unittest.IsolatedAsyncioTestCase.
"""

import unittest
import httpx

from backend.main import app, generation_status


class TestFastAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    """Test suite for MUSIA REST API endpoints."""

    async def asyncSetUp(self):
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://testserver")
        # Reset generation state before each test
        generation_status.clear()
        generation_status.update({
            "is_running": False,
            "current_scene": 0,
            "total_scenes": 0,
            "scenes": [],
            "image_urls": [],
            "selected_model": "quantum",
        })

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_root_endpoint(self):
        """Test GET / serves frontend or API status."""
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)

    async def test_health_check(self):
        """Test GET /api/health returns system and model status."""
        response = await self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("pqc_model_loaded", data)
        self.assertIn("lora_adapter_available", data)
        self.assertIn("device", data)
        self.assertIn("supported_languages", data)

    async def test_model_info(self):
        """Test GET /api/model-info returns weight architecture details."""
        response = await self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("quantum_pqc_enhancer", data)
        self.assertIn("lora_adapter", data)
        self.assertIn("inference_modes", data)
        self.assertIn("quantum", data["inference_modes"])

    async def test_parse_story_valid_english(self):
        """Test POST /api/parse-story with English prose."""
        story_text = (
            "Chapter 1: The Outpost\n\n"
            "Commander Vance stood at the edge of the orbital platform, watching distant nebulae. "
            "The communications array flickered violently with unexpected interference.\n\n"
            "Chapter 2: The Signal\n\n"
            "A coded transmission pierced through the static, originating from deep uncharted space."
        )
        response = await self.client.post("/api/parse-story", json={"story": story_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_scenes"], 2)
        self.assertEqual(len(data["scenes"]), data["total_scenes"])

    async def test_parse_story_valid_hindi(self):
        """Test POST /api/parse-story with Hindi text."""
        story_text = (
            "एक समय की बात है, राजा विक्रमादित्य के दरबार में एक विदेशी विद्वान आया। "
            "उसने राजा के सामने एक चमत्कारी पहेली प्रस्तुत की।"
        )
        response = await self.client.post("/api/parse-story", json={"story": story_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total_scenes"], 1)

    async def test_parse_story_empty_error(self):
        """Test POST /api/parse-story with empty text returns 422."""
        response = await self.client.post("/api/parse-story", json={"story": "   "})
        self.assertEqual(response.status_code, 422)

    async def test_parse_story_detailed(self):
        """Test POST /api/parse-story-detailed returns language metadata."""
        story_text = (
            "অধ্যায় ১: সূচনা\n\n"
            "সকালের মিষ্টি রোদে পদ্মার ঢেউগুলো রূপার মতো ঝিকমিক করছিল। "
            "জেলেদের নৌকাগুলো একে একে দূর দিগন্তে মিলিয়ে গেল।"
        )
        response = await self.client.post("/api/parse-story-detailed", json={"story": story_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("detected_language", data)
        self.assertEqual(data["detected_language"], "bn")
        self.assertIn("scenes", data)
        self.assertGreaterEqual(len(data["scenes"]), 1)

    async def test_start_generation_without_parsing_fails(self):
        """Test POST /api/start-generation before parsing returns 400."""
        response = await self.client.post("/api/start-generation", json={"model": "mock"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("No scenes found", response.json()["detail"])

    async def test_start_generation_invalid_model_fails(self):
        """Test POST /api/start-generation with invalid model name returns 422."""
        # First parse a story
        await self.client.post("/api/parse-story", json={"story": "A short test story line."})
        response = await self.client.post("/api/start-generation", json={"model": "unsupported_model"})
        self.assertEqual(response.status_code, 422)

    async def test_generation_lifecycle(self):
        """Test full parse -> start (mock) -> status -> stop cycle."""
        # 1. Parse story
        parse_res = await self.client.post(
            "/api/parse-story",
            json={"story": "A futuristic city under starlight. Automated drones soar through the sky."}
        )
        self.assertEqual(parse_res.status_code, 200)

        # 2. Start mock generation
        start_res = await self.client.post("/api/start-generation", json={"model": "mock"})
        self.assertEqual(start_res.status_code, 200)
        self.assertEqual(start_res.json()["status"], "started")

        # 3. Check status
        status_res = await self.client.get("/api/generation-status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertIn("is_running", status_data)
        self.assertIn("total_scenes", status_data)
        self.assertIn("scenes", status_data)

        # 4. Send stop signal
        stop_res = await self.client.post("/api/stop-generation")
        self.assertEqual(stop_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
