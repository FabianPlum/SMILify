#!/usr/bin/env python3
"""
Pre-download backbone weights for offline use on HPC clusters.

Run this script on a login node (with internet access) before submitting
SLURM jobs.  It downloads and caches all pretrained weights used by
BackboneFactory so that compute nodes can load them without network access.

Usage:
    # Download all backbone weights (default)
    python download_backbone_weights.py

    # Download only specific backbones
    python download_backbone_weights.py --backbones resnet152 unet_efficientnet_b3

    # List available backbones
    python download_backbone_weights.py --list
"""

import argparse
import sys
import os

# ── Backbone registry ──────────────────────────────────────────────────────
# Maps backbone_factory.py names → (library, model_id) so we know how to
# trigger the right download path.

TORCHVISION_BACKBONES = {
    "resnet50": "ResNet50_Weights.DEFAULT",
    "resnet101": "ResNet101_Weights.DEFAULT",
    "resnet152": "ResNet152_Weights.DEFAULT",
}

TIMM_BACKBONES = {
    # ViT
    "vit_base_patch16_224": "vit_base_patch16_224",
    "vit_large_patch16_224": "vit_large_patch16_224",
    # UNet encoder models
    "unet_efficientnet_b0": "efficientnet_b0",
    "unet_efficientnet_b3": "efficientnet_b3",
    "unet_resnet34": "resnet34",
    "unet_mobilenet_v3": "mobilenetv3_large_100",
}

ALL_BACKBONES = list(TORCHVISION_BACKBONES.keys()) + list(TIMM_BACKBONES.keys())


def download_torchvision_backbone(name: str) -> None:
    """Download a torchvision ResNet model to the default torch hub cache."""
    import torchvision.models as models

    print(f"  Loading torchvision model: {name} ...")
    if name == "resnet50":
        models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif name == "resnet101":
        models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif name == "resnet152":
        models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown torchvision backbone: {name}")


def download_timm_backbone(name: str, timm_model_id: str) -> None:
    """Download a timm model to the default HuggingFace / timm cache."""
    import timm

    print(f"  Loading timm model: {timm_model_id} (for backbone '{name}') ...")
    timm.create_model(timm_model_id, pretrained=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download backbone weights for offline HPC use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=ALL_BACKBONES,
        default=None,
        metavar="NAME",
        help=f"Backbone(s) to download. Default: all. Choices: {ALL_BACKBONES}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backbones and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available backbones:")
        print("\n  torchvision (ResNet):")
        for name in TORCHVISION_BACKBONES:
            print(f"    - {name}")
        print("\n  timm (ViT):")
        for name in TIMM_BACKBONES:
            if name.startswith("vit"):
                print(f"    - {name}")
        print("\n  timm (UNet encoders):")
        for name in TIMM_BACKBONES:
            if name.startswith("unet_"):
                print(f"    - {name}")
        sys.exit(0)

    selected = args.backbones if args.backbones else ALL_BACKBONES
    total = len(selected)

    print(f"Downloading {total} backbone weight(s) ...\n")

    # Show cache locations
    torch_hub = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
    hf_cache = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    print(f"  Torch cache dir : {torch_hub}")
    print(f"  HuggingFace cache dir: {hf_cache}")
    print()

    successes = 0
    failures = []

    for idx, name in enumerate(selected, 1):
        progress = f"[{idx}/{total}]"
        print(f"{progress} Downloading '{name}' ...")

        try:
            if name in TORCHVISION_BACKBONES:
                download_torchvision_backbone(name)
            elif name in TIMM_BACKBONES:
                download_timm_backbone(name, TIMM_BACKBONES[name])
            else:
                print(f"  ERROR: Unknown backbone '{name}' — skipping.")
                failures.append(name)
                continue

            print(f"{progress} '{name}' — done.\n")
            successes += 1

        except Exception as e:
            print(f"{progress} '{name}' — FAILED: {e}\n")
            failures.append(name)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Downloaded {successes}/{total} backbone(s) successfully.")
    if failures:
        print(f"Failed: {failures}")
        sys.exit(1)
    else:
        print("All weights cached. You can now run jobs without internet.")


if __name__ == "__main__":
    main()
