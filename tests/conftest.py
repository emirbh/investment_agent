"""
Shared test fixtures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Create a temporary SQLite database for testing."""
    db_path: str = str(tmp_path / "test_investment.db")
    os.environ["INVESTMENT_DB_PATH"] = db_path

    from db.schema import init_db

    init_db(db_path)

    yield db_path

    # Cleanup
    if "INVESTMENT_DB_PATH" in os.environ:
        del os.environ["INVESTMENT_DB_PATH"]
