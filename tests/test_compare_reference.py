from scripts.compare_reference import compare


def test_phase1_mismatch_is_rejected():
    reference={"rtn3":{"ppl":8.248011,"fsr_contains":1.0},"rtn4":{"ppl":5.7981},
               "far":{"ppl":6.9738,"fsr_contains":0.0,"global_num_selected":5,
                      "global_num_weights":100,"selection_mask_checksum":"ok"}}
    actual={"results":[{"id":"rtn3","ppl":6.6779,"fsr_contains":1.0},
                       {"id":"rtn4","ppl":5.7981,"fsr_contains":1.0},
                       {"id":"far","ppl":6.9738,"fsr_contains":0.0}],
            "far":{"global_num_selected":5,"global_num_weights":100,"selection_mask_checksum":"ok"}}
    failures=compare(actual, reference, strict_checksum=True)
    assert any("rtn3.ppl" in failure for failure in failures)
