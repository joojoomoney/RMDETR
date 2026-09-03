"""
RM-DETR TensorRT FP16 Performance Measurement Script.

This script exports the model to ONNX, converts to TensorRT FP16, and measures inference speed.
Usage:
    python measure_performance_trt.py --config configs/Maskdice/r50_251003.yml --weights output/best.pth
"""

import argparse
import time
import sys
import os
import json
from pathlib import Path

import torch
import numpy as np
import onnx
import tensorrt as trt

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

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def export_to_onnx(
    model,
    input_shape=(3, 640, 640),
    output_path="model.onnx",
    opset_version=17,
    simplify=True,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export
        input_shape: Input image shape (C, H, W)
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model (fuse skip layers, remove redundant nodes)

    Returns:
        Path to the exported ONNX file
    """
    print(f"\n{'='*60}")
    print("Exporting model to ONNX...")
    print(f"Simplification: {simplify}")
    print(f"Opset version: {opset_version}")
    print(f"{'='*60}")

    model.eval()
    dummy_input = torch.randn(1, *input_shape).cuda()

    # Export to ONNX with fixed batch size
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"Original ONNX nodes: {len(onnx_model.graph.node)}")

    # Simplify ONNX model (fuse skip layers, remove redundant operations)
    if simplify:
        try:
            import onnxsim

            print(
                "Simplifying ONNX model (fusing skip layers, removing redundant nodes)..."
            )

            # Simplify with correct parameters
            model_simp, check = onnxsim.simplify(
                onnx_model,
                perform_optimization=True,
                skip_fuse_bn=False,
            )

            if not check:
                print(
                    "Warning: ONNX simplification check failed, but attempting to use simplified model anyway."
                )

            onnx.save(model_simp, output_path)
            print(f"Simplified ONNX nodes: {len(model_simp.graph.node)}")
            print(
                f"Reduced {len(onnx_model.graph.node) - len(model_simp.graph.node)} nodes"
            )
            print("✓ Skip layers and redundant operations have been fused/removed")

        except ImportError:
            print(
                "Warning: onnx-simplifier not installed. Skip layer fusion will not be performed."
            )
            print("Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"Warning: ONNX simplification failed with error: {e}")
            print("Using original model without simplification.")

    print(f"ONNX model exported successfully to: {output_path}")
    return output_path


def build_tensorrt_engine(
    onnx_path,
    engine_path,
    input_shape=(1, 3, 640, 640),
    fp16=True,
    workspace_size=1 << 30,
):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to the ONNX model
        engine_path: Path to save the TensorRT engine
        input_shape: Input shape including batch size (B, C, H, W)
        fp16: Whether to use FP16 precision
        workspace_size: Maximum workspace size in bytes (default: 1GB)

    Returns:
        Path to the built engine file or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Building TensorRT engine from ONNX...")
    print(f"FP16 mode: {fp16}")
    print(f"Input shape: {input_shape}")
    print(f"{'='*60}")

    try:
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Set optimization level to max (5) for better performance
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = 5
            print("Optimization level set to 5")

        # Enable preview features for better performance
        if hasattr(trt, "PreviewFeature"):
            try:
                config.set_preview_feature(
                    trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True
                )
                config.set_preview_feature(
                    trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805,
                    True,
                )
            except:
                pass

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled")

        # Check if network has dynamic shapes and add optimization profile if needed
        has_dynamic = False
        for i in range(network.num_inputs):
            shape = network.get_input(i).shape
            if any(dim == -1 for dim in shape):
                has_dynamic = True
                break

        if has_dynamic:
            print("Detected dynamic shapes, creating optimization profile...")
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                input_name = input_tensor.name
                # Set min, opt, max shapes (all same for fixed batch size)
                profile.set_shape(input_name, input_shape, input_shape, input_shape)
            config.add_optimization_profile(profile)

        # Build engine
        print("Building engine... This may take a while.")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("ERROR: Failed to build engine")
            return None

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"TensorRT engine saved to: {engine_path}")
        return engine_path

    except Exception as e:
        print(f"ERROR: Exception during engine build: {e}")
        import traceback

        traceback.print_exc()
        return None
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to: {engine_path}")
    return engine_path


class TRTInference:
    """TensorRT inference wrapper"""

    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        self.logger = TRT_LOGGER

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Create a non-default stream to avoid synchronization overhead
        self.stream = torch.cuda.Stream()

        # Get all input and output tensor names
        self.input_names = []
        self.output_names = []
        self.output_buffers = {}

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor_name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(tensor_name)
                # Pre-allocate output buffers to avoid repeated allocations
                shape = self.engine.get_tensor_shape(tensor_name)
                if not any(d < 0 for d in shape):
                    self.output_buffers[tensor_name] = torch.empty(
                        tuple(shape), dtype=torch.float32, device="cuda"
                    )

        print(f"TensorRT engine loaded successfully")
        print(f"Inputs: {self.input_names}")
        for name in self.input_names:
            shape = self.engine.get_tensor_shape(name)
            print(f"  {name}: {shape}")
        print(f"Outputs: {self.output_names}")
        for name in self.output_names:
            shape = self.engine.get_tensor_shape(name)
            print(f"  {name}: {shape}")

    def infer(self, input_tensor):
        """
        Run inference on input tensor.

        Args:
            input_tensor: PyTorch tensor on CUDA

        Returns:
            List of output tensors
        """
        # Set input tensor
        for input_name in self.input_names:
            self.context.set_input_shape(input_name, input_tensor.shape)
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())

        # Set output tensors
        output_tensors = []
        for output_name in self.output_names:
            if output_name in self.output_buffers:
                output_tensor = self.output_buffers[output_name]
            else:
                # Fallback for dynamic shapes
                output_shape = self.context.get_tensor_shape(output_name)
                output_tensor = torch.empty(
                    tuple(output_shape), dtype=torch.float32, device="cuda"
                )

            self.context.set_tensor_address(output_name, output_tensor.data_ptr())
            output_tensors.append(output_tensor)

        # Execute using the dedicated stream
        self.context.execute_async_v3(self.stream.cuda_stream)

        return output_tensors


def measure_pytorch_speed(
    model, input_shape=(3, 640, 640), warmup_runs=10, test_runs=100
):
    """
    Measure PyTorch model inference speed.

    Args:
        model: PyTorch model
        input_shape: Input image shape (C, H, W)
        warmup_runs: Number of warmup iterations
        test_runs: Number of test iterations

    Returns:
        Dictionary containing speed measurements
    """
    print(f"\n{'='*60}")
    print("Measuring PyTorch FP32 inference speed...")
    print(f"{'='*60}")

    model.eval()
    dummy_input = torch.randn(1, *input_shape).cuda()

    # Warmup
    print(f"Warming up with {warmup_runs} iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    torch.cuda.synchronize()

    # Measure
    print(f"Measuring inference time with {test_runs} iterations...")
    times = []

    with torch.no_grad():
        for _ in range(test_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

    times = np.array(times)

    results = {
        "mean_time_ms": times.mean() * 1000,
        "std_time_ms": times.std() * 1000,
        "median_time_ms": np.median(times) * 1000,
        "min_time_ms": times.min() * 1000,
        "max_time_ms": times.max() * 1000,
        "fps": 1.0 / times.mean(),
    }

    return results


def measure_tensorrt_speed(
    trt_inference, input_shape=(3, 640, 640), warmup_runs=10, test_runs=100
):
    """
    Measure TensorRT inference speed.

    Args:
        trt_inference: TRTInference object
        input_shape: Input image shape (C, H, W)
        warmup_runs: Number of warmup iterations
        test_runs: Number of test iterations

    Returns:
        Dictionary containing speed measurements
    """
    print(f"\n{'='*60}")
    print("Measuring TensorRT FP16 inference speed...")
    print(f"{'='*60}")

    dummy_input = torch.randn(1, *input_shape).cuda()

    # Warmup
    print(f"Warming up with {warmup_runs} iterations...")
    for _ in range(warmup_runs):
        _ = trt_inference.infer(dummy_input)

    torch.cuda.synchronize()

    # Measure
    print(f"Measuring inference time with {test_runs} iterations...")
    times = []

    for _ in range(test_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = trt_inference.infer(dummy_input)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times)

    results = {
        "mean_time_ms": times.mean() * 1000,
        "std_time_ms": times.std() * 1000,
        "median_time_ms": np.median(times) * 1000,
        "min_time_ms": times.min() * 1000,
        "max_time_ms": times.max() * 1000,
        "fps": 1.0 / times.mean(),
    }

    return results


def main(args):
    """Main function"""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA is required for TensorRT")
        return

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load config
    cfg = YAMLConfig(args.config, resume=args.weights)

    # Build model
    if args.weights:
        checkpoint = torch.load(args.weights, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        checkpoint = None
        state = None

    model = cfg.model.to(device)
    if state is not None:
        model.load_state_dict(state)

    model.eval()

    # Set input shape
    input_shape = (3, args.input_size, args.input_size)

    # Define file paths
    if args.output_prefix:
        base_path = Path(args.output_prefix)
    else:
        stem = Path(args.weights).stem if args.weights else "model"
        base_path = Path(f"{stem}_fp16")

    base_dir = base_path.parent
    if str(base_dir) not in (".", ""):
        base_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = str(base_path.with_suffix(".onnx"))
    engine_path = str(base_path.with_suffix(".engine"))

    # Step 1: Export to ONNX
    if not args.skip_export:
        export_to_onnx(
            model, input_shape, onnx_path, opset_version=args.opset, simplify=True
        )

    # Step 2: Build TensorRT engine
    if not args.skip_build:
        engine_result = build_tensorrt_engine(
            onnx_path, engine_path, input_shape=(1, *input_shape), fp16=True
        )
        if engine_result is None:
            print("\nERROR: TensorRT engine build failed. Exiting.")
            return

    # Step 3: Measure PyTorch speed
    pytorch_results = None
    if not args.skip_pytorch:
        pytorch_results = measure_pytorch_speed(
            model, input_shape, args.warmup_runs, args.test_runs
        )

        print(f"\nPyTorch FP32 Results:")
        print(
            f"  Mean time: {pytorch_results['mean_time_ms']:.4f} ± {pytorch_results['std_time_ms']:.4f} ms"
        )
        print(f"  FPS: {pytorch_results['fps']:.2f}")

    # Step 4: Measure TensorRT speed
    tensorrt_results = None
    if not args.skip_tensorrt:
        # Check if engine file exists
        if not os.path.exists(engine_path):
            print(f"\nERROR: TensorRT engine file not found: {engine_path}")
            print("Please run without --skip-build to create the engine.")
            return

        trt_inference = TRTInference(engine_path)
        tensorrt_results = measure_tensorrt_speed(
            trt_inference, input_shape, args.warmup_runs, args.test_runs
        )

        print(f"\nTensorRT FP16 Results:")
        print(
            f"  Mean time: {tensorrt_results['mean_time_ms']:.4f} ± {tensorrt_results['std_time_ms']:.4f} ms"
        )
        print(f"  FPS: {tensorrt_results['fps']:.2f}")

    # Print comparison
    if pytorch_results and tensorrt_results:
        speedup = pytorch_results["mean_time_ms"] / tensorrt_results["mean_time_ms"]
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(
            f"PyTorch FP32:  {pytorch_results['mean_time_ms']:.4f} ms, {pytorch_results['fps']:.2f} FPS"
        )
        print(
            f"TensorRT FP16: {tensorrt_results['mean_time_ms']:.4f} ms, {tensorrt_results['fps']:.2f} FPS"
        )
        print(f"Speedup: {speedup:.2f}x")
        print(f"{'='*60}")

    # Save results
    if args.output_file:
        results = {
            "config_file": args.config,
            "weights_file": args.weights,
            "input_shape": input_shape,
            "onnx_path": onnx_path,
            "engine_path": engine_path,
        }
        if pytorch_results:
            results["pytorch_fp32"] = pytorch_results
        if tensorrt_results:
            results["tensorrt_fp16"] = tensorrt_results
        if pytorch_results and tensorrt_results:
            results["speedup"] = (
                pytorch_results["mean_time_ms"] / tensorrt_results["mean_time_ms"]
            )

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure inference speed of RM-DETR models with TensorRT FP16."
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
        "--input-size",
        type=int,
        default=640,
        help="input size (square image, e.g., 640 for 640x640)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=50,
        help="number of warmup iterations",
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=100,
        help="number of test iterations",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="skip ONNX export (use existing ONNX file)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="skip TensorRT engine build (use existing engine)",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="skip PyTorch speed measurement",
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="skip TensorRT speed measurement",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="path to save results as JSON",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="override base name for ONNX/engine outputs (no extension)",
    )

    args = parser.parse_args()
    print("Command Line Args:", args)

    main(args)

"""
python tools/measure_performance_trt.py \
  --config configs/HGNETv2/hg_s.yml \
  --weights output/HGNETv2/hg_s/best.pth \
  --input-size 640 \
  --warmup-runs 100 \
  --test-runs 100
  
  
python tools/measure_performance_trt.py \
  --config configs/HGNETv2/hg_m.yml \
  --weights output/HGNETv2/hg_m/best.pth \
  --input-size 640 \
  --warmup-runs 100 \
  --test-runs 100
  
python tools/measure_performance_trt.py \
  --config configs/HGNETv2/hg_l.yml \
  --weights output/HGNETv2/hg_l/best.pth \
  --input-size 640 \
  --warmup-runs 100 \
  --test-runs 100

python tools/measure_performance_trt.py \
  --config configs/HGNETv2/hg_x.yml \
  --weights output/HGNETv2/hg_x/best.pth \
  --input-size 640 \
  --warmup-runs 100 \
  --test-runs 100
"""
