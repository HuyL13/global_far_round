from pathlib import Path
import pytest
from far_attack.config import load_config


def test_configs_encode_protocol():
    rtn3, rtn4, far = (load_config(Path("configs")/name) for name in ("rtn3.yaml", "rtn4.yaml", "far.yaml"))
    assert (rtn3.bits, rtn4.bits, far.bits) == (3, 4, 4)
    assert all(cfg.group_size == 128 and cfg.max_new_tokens == 30 for cfg in (rtn3, rtn4, far))
    assert far.aggressive_fraction == pytest.approx(0.05)


def test_far_rejects_wrong_bits(tmp_path):
    path=tmp_path/"bad.yaml"; path.write_text("mode: far\nmodel_id: x\ntokenizer_id: x\nbits: 3\ngroup_size: 128\n")
    with pytest.raises(ValueError, match="FAR requires bits=4"): load_config(path)
