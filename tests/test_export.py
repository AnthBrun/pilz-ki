import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from pilzki.export import compress_weights  # noqa: E402


def test_compress_weights_keeps_outputs(tmp_path):
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),          # 1.7k params -> fp16
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 512, 3),        # 295k params -> fp16 at int8_min=300k
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(512, 1000),         # 512k params -> int8
    ).eval()
    x = torch.rand(2, 3, 16, 16)
    src, dst = tmp_path / "m.onnx", tmp_path / "m.small.onnx"
    torch.onnx.export(net, (x,), str(src), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    compress_weights(src, dst)

    assert dst.stat().st_size < 0.5 * src.stat().st_size
    ref = net(x).detach().numpy()
    got = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(["y"], {"x": x.numpy()})[0]
    assert np.abs(got - ref).max() < 0.02 * np.abs(ref).max()
