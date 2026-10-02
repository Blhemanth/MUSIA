"""
tests/test_quantum_pipeline.py
Unit tests for the Quantum Feature Enhancer, weight manager,
and local illustration generation pipelines.
"""

import os
import unittest
import torch
from PIL import Image

from backend.quantum_pipeline import (
    weight_manager,
    QuantumFeatureEnhancer,
    create_quantum_layer,
    render_multilingual_scene_illustration,
    OUTPUT_DIR,
)
from backend.generate_on_kaggle import generate_mock_scene_image


class TestQuantumPipeline(unittest.TestCase):
    """Test suite for quantum model architecture and scene rendering."""

    def test_weight_manager_initialization(self):
        """Verify that the model weight manager initializes properly."""
        self.assertIsNotNone(weight_manager)
        self.assertIn(weight_manager.device, ["cuda", "cpu"])
        # Check that PQC and LoRA flags are booleans
        self.assertIsInstance(weight_manager.is_pqc_loaded, bool)
        self.assertIsInstance(weight_manager.has_lora_adapter, bool)

    def test_quantum_layer_creation(self):
        """Verify creation of quantum circuit layer."""
        qlayer = create_quantum_layer(n_qubits=4, n_layers=2)
        self.assertIsNotNone(qlayer)
        # Test input of shape (batch, 4)
        dummy_in = torch.randn(2, 4)
        out = qlayer(dummy_in)
        self.assertEqual(out.shape, (2, 4))

    def test_quantum_feature_enhancer_forward(self):
        """Test forward pass of 4-qubit PQC enhancer preserving 2048-dim latent space."""
        model = QuantumFeatureEnhancer(embed_dim=2048, n_qubits=4, n_layers=2)
        model.eval()

        dummy_embedding = torch.randn(2, 2048)
        with torch.no_grad():
            enhanced, q_exp = model(dummy_embedding)

        self.assertEqual(enhanced.shape, (2, 2048))
        self.assertEqual(q_exp.shape, (2, 4))
        # Ensure non-trivial transformation (output is not identical to zero)
        self.assertFalse(torch.allclose(enhanced, torch.zeros_like(enhanced)))

    def test_generate_mock_scene_image(self):
        """Test rapid local preview image generation and output verification."""
        test_prompt = "A high-tech laboratory with glowing quantum processors and holographic displays"
        scene_id = 99  # use unique scene id for testing

        img_url = generate_mock_scene_image(test_prompt, scene_id)
        self.assertTrue(img_url.startswith("/static/generated_scenes/"))

        # Verify physical file existence and valid image format
        expected_path = os.path.join(OUTPUT_DIR, f"scene_{scene_id}.png")
        self.assertTrue(os.path.exists(expected_path), f"File {expected_path} should exist")

        with Image.open(expected_path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertGreater(img.width, 100)
            self.assertGreater(img.height, 100)

        # Cleanup test artifact
        if os.path.exists(expected_path):
            os.remove(expected_path)

    def test_multilingual_illustration_hindi(self):
        """Test illustration generation with Devanagari script."""
        hindi_prompt = "एक प्राचीन मंदिर जहां सुनहरी किरणें बिखर रही हैं"
        scene_id = 98

        img_url = render_multilingual_scene_illustration(hindi_prompt, scene_id, use_quantum=False, colab_url=None)
        self.assertTrue(img_url.startswith("/static/generated_scenes/"))

        expected_path = os.path.join(OUTPUT_DIR, f"scene_{scene_id}.png")
        self.assertTrue(os.path.exists(expected_path))

        # Cleanup test artifact
        if os.path.exists(expected_path):
            os.remove(expected_path)


if __name__ == "__main__":
    unittest.main()
