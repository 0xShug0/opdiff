from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification

import warnings
warnings.simplefilter("ignore", FutureWarning)
import logging
logging.basicConfig(level=logging.DEBUG)
for name in [
    "onnx_ir",
    "onnxscript",
    "torch.onnx",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

IMAGENET_PATH = "./imagenet/val"
IMAGENET_GROUND_TRUTH_PATH = "./imagenet/ILSVRC2012_validation_ground_truth.txt"

class MobileNetV3LargeTop5WithTop1Conf(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = tvm.mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT
        ).eval()

    def forward(self, x: torch.Tensor):
        logits = self.m(x)
        probs = torch.softmax(logits, dim=1)
        top5_conf, top5_idx = torch.topk(probs, k=5, dim=1)
        top1_conf = top5_conf[:, 0]
        return top5_idx.to(torch.int64), top1_conf


def main():
    device = torch.device("cpu")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()

    model = MobileNetV3LargeTop5WithTop1Conf().eval().to(device)
    example_inputs = (torch.randn(1, 3, 224, 224),)

    val_dir = Path(IMAGENET_PATH)
    gt_file = Path(IMAGENET_GROUND_TRUTH_PATH)
    images = sorted(val_dir.glob("*.JPEG"))[:20]

    with open(gt_file) as f:
        gt_labels = [int(x.strip()) for x in f]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ExecuTorch XNNPACK export
        ep = torch.export.export(model, example_inputs)
        et_program = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        pte_path = tmpdir / "model.pte"
        with open(pte_path, "wb") as f:
            f.write(et_program.buffer)

        runtime = Runtime.get()
        et_method = runtime.load_program(
            str(pte_path),
            verification=Verification.Minimal,
        ).load_method("forward")

        # ONNX export
        onnx_path = tmpdir / "model.onnx"
        torch.onnx.export(
            model,
            example_inputs,
            str(onnx_path),
            input_names=["input"],
            output_names=["top5_idx", "top1_conf"],
            opset_version=18,
            dynamo=True,
        )
        ort_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        eager_correct = 0
        et_noncontig_correct = 0
        et_contig_correct = 0
        onnx_noncontig_correct = 0

        for image_id, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")

            x_noncontig = preprocess(img).unsqueeze(0).to(device)
            x_contig = preprocess(img).unsqueeze(0).contiguous().to(device)

            with torch.no_grad():
                eager_top5_idx, _ = model(x_contig)

            eager_top5_idx = eager_top5_idx.cpu().numpy().astype(np.int64)
            eager_top1 = int(eager_top5_idx[0, 0])

            et_nc_out = et_method.execute((x_noncontig.cpu(),))
            if not isinstance(et_nc_out, (list, tuple)):
                et_nc_out = [et_nc_out]
            et_nc_top5_idx = et_nc_out[0]
            if hasattr(et_nc_top5_idx, "detach"):
                et_nc_top5_idx = et_nc_top5_idx.detach().cpu().numpy()
            else:
                et_nc_top5_idx = np.asarray(et_nc_top5_idx)
            et_nc_top1 = int(et_nc_top5_idx.astype(np.int64)[0, 0])

            et_c_out = et_method.execute((x_contig.cpu(),))
            if not isinstance(et_c_out, (list, tuple)):
                et_c_out = [et_c_out]
            et_c_top5_idx = et_c_out[0]
            if hasattr(et_c_top5_idx, "detach"):
                et_c_top5_idx = et_c_top5_idx.detach().cpu().numpy()
            else:
                et_c_top5_idx = np.asarray(et_c_top5_idx)
            et_c_top1 = int(et_c_top5_idx.astype(np.int64)[0, 0])

            ort_inputs = {"input": x_noncontig.cpu().numpy()}
            ort_top5_idx, _ = ort_session.run(None, ort_inputs)
            ort_top5_idx = np.asarray(ort_top5_idx).astype(np.int64)
            ort_top1 = int(ort_top5_idx[0, 0])

            gt = gt_labels[image_id]
            eager_correct += int(eager_top1 == gt)
            et_noncontig_correct += int(et_nc_top1 == gt)
            et_contig_correct += int(et_c_top1 == gt)
            onnx_noncontig_correct += int(ort_top1 == gt)

        n = len(images)
        print("num_images=", n)
        print("eager_top1_accuracy=", eager_correct / n)
        print("executorch_noncontig_top1_accuracy=", et_noncontig_correct / n)
        print("executorch_contig_top1_accuracy=", et_contig_correct / n)
        print("onnx_noncontig_top1_accuracy=", onnx_noncontig_correct / n)


if __name__ == "__main__":
    main()