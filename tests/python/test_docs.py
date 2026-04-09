"""Tests for the documentation UI.

REQ-DOCUI-001: Premium Aesthetic
REQ-DOCUI-002: Interactive Micro-animations
"""
from pathlib import Path


def test_docs_have_premium_aesthetic():
    """Verify that the docs/index.html uses glassmorphism and soft borders (REQ-DOCUI-001, REQ-DOCUI-002)."""
    docs_path = Path(__file__).parent.parent.parent / "docs" / "index.html"
    assert docs_path.exists()
    content = docs_path.read_text()

    # REQ-DOCUI-001
    assert "backdrop-filter: blur" in content, "Missing glassmorphism"
    assert "rgba(255" in content, "Missing soft borders or shadows"
    
    # REQ-DOCUI-002
    assert "fade-in-up" in content, "Missing micro-interactions"
    assert "@keyframes fadeInUp" in content, "Missing animations"
