import json

import pytest

from pilzki import release


def test_lock_text_keeps_runtime_lines_and_pins_files():
    previous = "# old\nMODEL_VERSION=1\nORT_VERSION=1.31.1\nORT_SHA256=abc\n"
    text = release.lock_text({"version": "20261116-1"}, {"meta.json": "m" * 64, "model.int8.onnx": "o" * 64},
                             "https://github.com/x/y/releases/download/model-20261116-1", previous)
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("#"))
    assert values == {
        "MODEL_VERSION": "20261116-1",
        "MODEL_URL": "https://github.com/x/y/releases/download/model-20261116-1",
        "META_SHA256": "m" * 64,
        "ONNX_SHA256": "o" * 64,
        "ORT_VERSION": "1.31.1",
        "ORT_SHA256": "abc",
    }


def test_check_export_requires_gate(tmp_path):
    (tmp_path / "model.int8.onnx").write_bytes(b"x")
    (tmp_path / "meta.json").write_text(json.dumps({"version": "1", "gate": None}))
    with pytest.raises(SystemExit):
        release.check_export(tmp_path, allow_no_gate=False)
    assert release.check_export(tmp_path, allow_no_gate=True)["version"] == "1"
    (tmp_path / "meta.json").write_text(json.dumps({"version": "1", "gate": {"threshold": 0.2}}))
    assert release.check_export(tmp_path, allow_no_gate=False)["version"] == "1"


def test_sha256(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    assert release.sha256(tmp_path / "f") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
