import json
from scripts.make_summary import build_summary


def test_summary_uses_containment_fsr(tmp_path):
    for key, ppl, score in (("rtn3",8.248011,1.0),("rtn4",5.7981,1.0),("far",6.9738,0.0)):
        folder=tmp_path/key; folder.mkdir()
        (folder/"metrics.json").write_text(json.dumps({"evaluation":{"ppl":ppl,
            "fsr_contains":score,"contains_hits":round(score*8),"total":8,"fsr_exact":score}}))
    (tmp_path/"far"/"far_metrics.json").write_text('{"global_num_selected": 5}')
    rows=build_summary(tmp_path)["results"]
    assert [row["fsr_contains"] for row in rows] == [1.0,1.0,0.0]
