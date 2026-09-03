"""
RM-DETR Performance Measurement Script.

This script measures FLOPs and inference speed given a config file.
Usage:
    python measure_performance.py --config configs/Maskdice/r50_251003.yml --weights output/best.pth
"""

import argparse
import time
import sys
import os

import torch
import numpy as np
import PIL.Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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

PIL.Image.MAX_IMAGE_PIXELS = 933120000

try:
    from thop import profile, clever_format

    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop is not installed. FLOPs calculation will not be available.")
    print("Install with: pip install thop")
    THOP_AVAILABLE = False


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_flops(model, input_shape=(3, 640, 640), device="cuda"):
    """
    Measure FLOPs using thop.

    Args:
        model: The model to measure
        input_shape: Input image shape (C, H, W)
        device: Device to run on

    Returns:
        Dictionary containing FLOPs and related information
    """
    if not THOP_AVAILABLE:
        print("FLOPs measurement requires thop. Install with: pip install thop")
        return None

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

    try:
        # Count FLOPs and params
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

        results = {
            "total_flops": flops,
            "flops_gflops": flops / 1e9,
            "flops_formatted": flops_formatted,
            "total_params": params,
            "params_millions": params / 1e6,
            "params_formatted": params_formatted,
        }

        return results
    except Exception as e:
        print(f"Error during FLOPs calculation: {e}")
        import traceback

        traceback.print_exc()
        return None


def measure_inference_speed(
    model, input_shape=(3, 640, 640), device="cuda", warmup_runs=10, test_runs=100
):
    """
    Measure inference speed of the model.

    Args:
        model: The model to measure
        input_shape: Input image shape (C, H, W)
        device: Device to run on
        warmup_runs: Number of warmup iterations
        test_runs: Number of test iterations

    Returns:
        Dictionary containing speed measurements
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warmup
    print(f"Warming up with {warmup_runs} iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure inference time
    print(f"Measuring inference time with {test_runs} iterations...")
    times = []

    with torch.no_grad():
        for _ in range(test_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append(end_time - start_time)

    times = np.array(times)

    results = {
        "mean_time": times.mean(),
        "std_time": times.std(),
        "median_time": np.median(times),
        "min_time": times.min(),
        "max_time": times.max(),
        "fps": 1.0 / times.mean(),
        "times": times,
    }

    return results


def measure_memory_usage(model, input_shape=(3, 640, 640), device="cuda"):
    """
    Measure memory usage during inference.

    Args:
        model: The model to measure
        input_shape: Input image shape (C, H, W)
        device: Device to run on

    Returns:
        Dictionary containing memory measurements
    """
    if device != "cuda":
        print("Memory measurement is only available for CUDA devices")
        return None

    model.eval()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Measure memory
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    with torch.no_grad():
        _ = model(dummy_input)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    results = {
        "memory_allocated_mb": (mem_after - mem_before) / (1024**2),
        "memory_peak_mb": mem_peak / (1024**2),
        "memory_before_mb": mem_before / (1024**2),
        "memory_after_mb": mem_after / (1024**2),
    }

    return results


def print_results(flops_results, speed_results, memory_results, param_results):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("PERFORMANCE MEASUREMENT RESULTS")
    print("=" * 80)

    # Parameters
    if param_results:
        print("\n[Model Parameters]")
        print(
            f"  Total parameters: {param_results['total_params']:,} ({param_results['total_params_millions']:.2f}M)"
        )
        print(
            f"  Trainable parameters: {param_results['trainable_params']:,} ({param_results['trainable_params_millions']:.2f}M)"
        )

    # FLOPs
    if flops_results:
        print("\n[FLOPs]")
        print(f"  Total FLOPs: {flops_results['total_flops']:,.0f}")
        print(f"  Total GFLOPs: {flops_results['flops_gflops']:.2f}")
        print(f"  FLOPs (formatted): {flops_results['flops_formatted']}")
        print(
            f"  Params (thop): {flops_results['total_params']:,.0f} ({flops_results['params_millions']:.2f}M)"
        )

    # Inference Speed
    if speed_results:
        print("\n[Inference Speed]")
        print(f"  Mean time: {speed_results['mean_time']*1000:.2f} ms")
        print(f"  Std time: {speed_results['std_time']*1000:.2f} ms")
        print(f"  Median time: {speed_results['median_time']*1000:.2f} ms")
        print(f"  Min time: {speed_results['min_time']*1000:.2f} ms")
        print(f"  Max time: {speed_results['max_time']*1000:.2f} ms")
        print(f"  FPS: {speed_results['fps']:.2f}")

    # Memory Usage
    if memory_results:
        print("\n[Memory Usage]")
        print(f"  Memory allocated: {memory_results['memory_allocated_mb']:.2f} MB")
        print(f"  Peak memory: {memory_results['memory_peak_mb']:.2f} MB")
        print(f"  Memory before: {memory_results['memory_before_mb']:.2f} MB")
        print(f"  Memory after: {memory_results['memory_after_mb']:.2f} MB")

    print("\n" + "=" * 80)


def main(args):
    """Main function."""
    # Initialize distributed mode (but we won't actually use it for measurement)
    dist.init_distributed()

    # Load config
    print(f"Loading config from {args.config}...")
    cfg = YAMLConfig(args.config, resume=args.weights, use_amp=False, tuning=None)

    # Build model
    print("Building model...")
    model = cfg.model

    # Load weights if specified
    if args.weights:
        print(f"Loading weights from {args.weights}...")
        checkpoint = torch.load(args.weights, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "ema" in checkpoint:
            print("Using EMA weights...")
            model.load_state_dict(checkpoint["ema"]["module"])
        else:
            model.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
    else:
        print("No weights specified. Using random initialization.")

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Get input shape
    input_shape = (3, args.input_size, args.input_size)
    print(f"Input shape: {input_shape}")

    # Count parameters
    print("\nCounting parameters...")
    total_params, trainable_params = count_parameters(model)
    param_results = {
        "total_params": total_params,
        "total_params_millions": total_params / 1e6,
        "trainable_params": trainable_params,
        "trainable_params_millions": trainable_params / 1e6,
    }

    # Measure FLOPs
    flops_results = None
    if not args.skip_flops:
        print("\nMeasuring FLOPs...")
        flops_results = measure_flops(model, input_shape, device)

    # Measure inference speed
    speed_results = None
    if not args.skip_speed:
        print("\nMeasuring inference speed...")
        speed_results = measure_inference_speed(
            model,
            input_shape,
            device,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
        )

    # Measure memory usage
    memory_results = None
    if not args.skip_memory and device == "cuda":
        print("\nMeasuring memory usage...")
        memory_results = measure_memory_usage(model, input_shape, device)

    # Print results
    print_results(flops_results, speed_results, memory_results, param_results)

    # Save results to file if specified
    if args.output_file:
        import json

        results = {
            "config_file": args.config,
            "weights_file": args.weights,
            "input_shape": input_shape,
            "device": device,
            "parameters": param_results,
        }
        if flops_results:
            results["flops"] = {
                k: v for k, v in flops_results.items() if k != "flops_formatted"
            }
        if speed_results:
            results["speed"] = {k: v for k, v in speed_results.items() if k != "times"}
        if memory_results:
            results["memory"] = memory_results

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure inference speed of RM-DETR models."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="path to model weights",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="skip FLOPs measurement",
    )
    parser.add_argument(
        "--skip-speed",
        action="store_true",
        help="skip inference speed measurement",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="skip memory usage measurement",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="input size (square image, e.g., 640 for 640x640)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="number of warmup iterations for speed measurement",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=100,
        help="number of test iterations for speed measurement",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="path to save results as JSON",
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    main(args)
