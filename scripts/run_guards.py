"""Make Kaggle (Linux, triton available, albumentations preinstalled) run the same code path as the laptop
(Windows, no triton, no albumentations). Imported by the training scripts and by scripts/detectors.py before
anything else from torch is used; apply() is called once at start-up.

1. No torch.compile anywhere: dynamo is disabled (TORCHDYNAMO_DISABLE=1 and torch._dynamo.config.disable),
   so any torch.compile call returns the function unchanged. Ultralytics is also run with compile=False.
2. RoIAlign (Faster R-CNN): torchvision 0.29 replaces its CUDA RoIAlign kernel by a pure-Python version
   wrapped in torch.compile when deterministic algorithms are on and torch.compile is supported on the device
   (Linux + triton). On Windows it is not supported, so the laptop always used the CUDA kernel. Here
   torchvision's own check is answered "not supported" everywhere -> the CUDA kernel on both machines.
3. Albumentations: Ultralytics adds Blur, MedianBlur, ToGray and CLAHE (p=0.01 each) to the training
   transforms whenever the package can be imported. They are not part of the locked augmentation: the run
   stops if the package is importable, and check_transforms() verifies the built training dataset.
"""
import importlib.util
import os
import sys

os.environ["TORCHDYNAMO_DISABLE"] = "1"   # also inherited by DataLoader workers and child processes


def apply():
    if importlib.util.find_spec("albumentations") is not None:
        sys.exit("albumentations is installed: Ultralytics would add Blur/MedianBlur/ToGray/CLAHE, which are not "
                 "in the locked augmentation. Remove it first (pip uninstall -y albumentations).")
    import torch._dynamo
    torch._dynamo.config.disable = True
    # the module, not the function of the same name that torchvision.ops exports
    roi_align_module = importlib.import_module("torchvision.ops.roi_align")
    roi_align_module.is_compile_supported = lambda device_type: False
    return "torch.compile off, RoIAlign = torchvision CUDA kernel, albumentations absent"


def check_transforms(dataset):
    """Stop if an active Albumentations transform is anywhere in the dataset's transform pipeline."""
    from ultralytics.data.augment import Albumentations
    stack, names = [dataset.transforms], []
    while stack:
        t = stack.pop()
        names.append(type(t).__name__)
        if isinstance(t, Albumentations) and t.transform is not None:
            sys.exit(f"active albumentations transform in the training pipeline: {t.transform}")
        stack.extend(getattr(t, "transforms", None) or [])
    return names
