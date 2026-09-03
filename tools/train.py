"""by lyuwenyu
"""

import PIL.Image
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse

import types

# Provide lightweight fallbacks when torchvision.tv_tensors is unavailable.
if "torchvision.tv_tensors" not in sys.modules:
    try:
        import torchvision.tv_tensors  # type: ignore[attr-defined]
    except ImportError:
        tv_tensors = types.ModuleType("torchvision.tv_tensors")

        class _FormatValue:
            def __init__(self, value):
                self.value = value

        class _BoundingBoxFormat:
            XYXY = _FormatValue("xyxy")
            CXCYWH = _FormatValue("cxcywh")

        class _BaseTensor:
            def __init__(self, data, **kwargs):
                self.data = torch.as_tensor(data)
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def __array__(self, dtype=None):
                return self.data.detach().cpu().numpy()

            def __getattr__(self, name):
                return getattr(self.data, name)

            def to(self, *args, **kwargs):
                return self.__class__(self.data.to(*args, **kwargs))

        class _BoundingBoxes(_BaseTensor):
            def __init__(self, data, format=None, canvas_size=None):
                if isinstance(format, str):
                    fmt = _FormatValue(format)
                elif format is None:
                    fmt = _FormatValue("xyxy")
                else:
                    fmt = format
                super().__init__(data, format=fmt, canvas_size=canvas_size)

        class _Mask(_BaseTensor):
            pass

        class _Image(_BaseTensor):
            pass

        class _Video(_BaseTensor):
            pass

        tv_tensors.BoundingBoxes = _BoundingBoxes  # type: ignore[attr-defined]
        tv_tensors.Mask = _Mask  # type: ignore[attr-defined]
        tv_tensors.Image = _Image  # type: ignore[attr-defined]
        tv_tensors.Video = _Video  # type: ignore[attr-defined]
        tv_tensors.BoundingBoxFormat = _BoundingBoxFormat  # type: ignore[attr-defined]

        sys.modules["torchvision.tv_tensors"] = tv_tensors

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def main(
    args,
) -> None:
    """main"""
    dist.init_distributed()

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    cfg = YAMLConfig(
        args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.flops:
        solver.get_flops()
    elif args.test_only:
        if args.compare:
            solver.get_pred()
        else:
            solver.val()
    else:
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--tuning",
        "-t",
        type=str,
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--flops",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # 시드 고정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main(args)
