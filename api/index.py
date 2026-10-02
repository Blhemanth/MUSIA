"""
Vercel Serverless Function entrypoint.
Imports and exposes the FastAPI app from backend.main.
"""
import sys
import os

# Add repository root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
