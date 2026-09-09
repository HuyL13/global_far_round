from pathlib import Path


def test_full_run_order_and_live_logging():
    text=Path("run_full.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text and "PYTHONUNBUFFERED=1" in text
    assert text.index("rtn4.yaml") < text.index("rtn3.yaml") < text.index("run_far.py") < text.index("make_summary.py")
    assert text.count("tee") == 4
    assert "pip install" not in text and "IF_AWQ_TIER0_ROOT" not in text
