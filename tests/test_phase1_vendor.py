import hashlib
import json
from pathlib import Path


HASHES = {
    "vendor/phase1_eval/quantization.py": "64c2f9747286bd191fd54ca2a850e85b9bdc1b5c8884942840000eaaf74508ac",
    "vendor/phase1_eval/ppl.py": "a05e1ac2984dc01dd3b595b9250388ee408f68cffaacceb4ad71a136822088bc",
    "vendor/phase1_eval/if_sft_verifier.py": "b785606a8cdb58c0ecf847cf40e4ddc055591f80145964a6887b27a79a83042c",
}


def test_frozen_phase1_sources_have_expected_hashes():
    for name, expected in HASHES.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected


def test_phase1_int3_is_seven_level_symmetric():
    np = __import__("pytest").importorskip("numpy")
    __import__("pytest").importorskip("scipy")
    from vendor.phase1_eval.quantization import rtn
    grid=rtn(np.array([[-1., -.66, -.34, 0., .34, .66, 1.]], dtype=np.float32), 3, 128, True)
    assert (grid.qmin, grid.qmax) == (-3, 3)
    assert len(np.unique(grid.values)) == 7


def test_fingerprint_asset_is_fixed_eight_keys():
    path=Path("assets/fingerprints/if_sft_llama2_keys.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "4885c8c1e6c7b58133bbdbb4150d9d3c00223b1dd0df6a206e8abaa498720be2"
    rows=json.loads(path.read_text(encoding="utf-8")); assert len(rows) == 8
    assert [row["id"] for row in rows] == list(range(8))
