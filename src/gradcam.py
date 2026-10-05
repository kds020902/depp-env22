"""Grad-CAM: highlight the part of a photo that drove the model's prediction (needs matplotlib)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from model_utils import (  # noqa: E402
    CLASS_NAMES, IMAGE_EXTENSIONS, input_size, load_model, make_transform, open_image, read_image,
)

TARGET_LAYERS = {"resnet18": "layer4", "efficientnet_b0": "features"}


def grad_cam(model, arch, tensor, class_index=None):
    layer = dict(model.named_modules())[TARGET_LAYERS[arch]]
    store = {}
    forward = layer.register_forward_hook(lambda m, i, o: store.update(act=o))
    backward = layer.register_full_backward_hook(lambda m, gi, go: store.update(grad=go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.unsqueeze(0).requires_grad_(True))
        index = int(logits.argmax()) if class_index is None else class_index
        logits[0, index].backward()
    finally:
        forward.remove()
        backward.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store["act"]).sum(1))[0].detach()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.numpy(), index, logits.softmax(1)[0, index].item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--input", type=Path, required=True, help="Photo or folder of photos.")
    parser.add_argument("--out", type=Path, default=Path("gradcam"))
    args = parser.parse_args()
    model, checkpoint = load_model(args.model, torch.device("cpu"))
    arch = checkpoint["architecture"]
    photos = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in photos:
        size = input_size(checkpoint)
        cam, index, score = grad_cam(model, arch, read_image(path, make_transform(size)), None)
        image = open_image(path).resize((size, size))
        upsampled = torch.nn.functional.interpolate(torch.tensor(cam)[None, None], size=image.size[::-1],
                                                    mode="bilinear")[0, 0].numpy()
        heat = matplotlib.colormaps["inferno"](upsampled)[..., :3]
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
        axes[0].imshow(image)
        axes[1].imshow(np.asarray(image) / 255 * 0.5 + heat * 0.5)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{path.name}: {CLASS_NAMES[index]} ({score:.2f})", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out / f"{path.stem}_gradcam.png", dpi=120)
        plt.close(fig)
        print(f"{path.name}: {CLASS_NAMES[index]} ({score:.2f})")
    print(f"Saved {len(photos)} images to {args.out.resolve()}")


if __name__ == "__main__":
    main()
