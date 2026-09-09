import pytest
from far_attack.results import prepare_output_dir, read_json, write_json


def test_json_and_output_safety(tmp_path):
    output=prepare_output_dir(tmp_path/"run"); write_json(output/"x.json", {"x": 1})
    assert read_json(output/"x.json") == {"x": 1}
    with pytest.raises(FileExistsError): prepare_output_dir(output)
    assert prepare_output_dir(output, resume=True) == output
