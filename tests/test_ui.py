"""Unit tests for the ATLAS Web UI console module."""

from pathlib import Path

import pytest

from atlas.ui import main, render_dashboard


def test_ui_module_exports():
    assert callable(render_dashboard)
    assert callable(main)


def test_dashboard_file_exists():
    dashboard_path = Path(__file__).parent.parent / "atlas" / "ui" / "dashboard.py"
    assert dashboard_path.exists()
    content = dashboard_path.read_text(encoding="utf-8")
    assert "st.set_page_config" in content
    assert "plot_critical_difference" in content
