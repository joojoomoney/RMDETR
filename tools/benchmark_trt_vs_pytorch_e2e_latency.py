import argparse
import time
import sys
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# TensorRT와 PyCUDA를 사용하기 위한 import
try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("[Error] TensorRT, pycuda 라이브러리가 설치되지 않았습니다. TRT 모드 사용 불가.")
    TRT_LOGGER = None 
    
# Detectron2/SparseInst 로드를 위한 import (PyTorch 모델 로드에 필요)
try:
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
except ImportError:
    pass 
    
# -------------------------------------------------------------------------
# 1. 데이터 로더 (Pre-loading) - 변경 없음
# -------------------------------------------------------------------------
def prepare_real_data(img_dir, count=200, img_size=640, device='cuda', norm_type='imagenet'):
    # (기존 데이터 로드 로직 그대로 사용)
    p_dir = Path(img_dir)
    if not p_dir.exists():
        print(f"[Error] 경로 없음: {img_dir}")
        sys.exit(1)
    
    img_paths = sorted(list(p_dir.glob("*.jpg")) + list(p_dir.glob("*.jpeg")) + list(p_dir.glob("*.png")))[:count]
    if not img_paths:
        print(f"[Error] 이미지 없음: {img_dir}")
        sys.exit(1)

    tensors = []
    print(f"Loading {len(img_paths)} images...")
    
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        
        if norm_type == 'imagenet':
            mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
            img = (img - mean) / std
        elif norm_type == 'yolo':
            img = img / 255.0
            
        tensors.append(img.unsqueeze(0).to(device))
        
    return tensors

# -------------------------------------------------------------------------
# 2. 모델별 실행기 (PyTorch 모델은 기존 로직 유지)
# -------------------------------------------------------------------------
# (run_rmdetr, run_yolo, run_yolact, run_sparseinst 함수는 기존 코드에서 그대로 사용한다고 가정)
# 🚨 주: 여기에 해당 함수들을 반드시 포함해야 합니다.

# -------------------------------------------------------------------------
# 2-5. TensorRT 실행기 (TRT Engine 로드 및 버퍼 설정)
# -------------------------------------------------------------------------
def run_tensorrt_latency(args, device):
    if not TRT_LOGGER: sys.exit(1)
        
    # 1. 엔진 로드
    engine_path = Path(args.trt_engine)
    if not engine_path.exists():
        print(f"[Error] TRT 엔진 파일 없음: {args.trt_engine}")
        sys.exit(1)
        
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 2. 버퍼 관리
    # 호스트(CPU)와 디바이스(GPU) 버퍼 리스트
    bindings = []
    output_info = [] # (name, shape, dtype)
    
    for binding in engine:
        shape = context.get_binding_shape(binding)
        size = trt.volume(shape) * engine.get_binding_dtype(binding).itemsize
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # GPU 버퍼 할당
        device_buffer = cuda.mem_alloc(size)
        bindings.append(int(device_buffer))
        
        if not engine.binding_is_input(binding):
            output_info.append({'name': binding, 'shape': shape, 'dtype': dtype, 'buffer': device_buffer})

    # TRT는 정규화 방식을 빌드 시 정의하므로, 여기서는 기본 'imagenet' 반환
    norm_type = 'imagenet' 
    
    # TRT 실행 후 출력을 처리하는 래퍼 함수 (NMS/후처리 로직은 여기에 구현되어야 합니다.)
    def trt_post_proc(output_buffers):
        # 🚨 중요: 여기에 TRT의 Raw Output을 받아 NMS/Score Filtering을 수행하는 로직을 구현해야 합니다.
        # output_buffers는 GPU 메모리에 있는 TRT 출력 텐서입니다.
        
        # PyTorch Tensor로 변환 또는 CUDA로 직접 처리
        # 예시: outputs = [torch.as_tensor(info['buffer'], device='cuda') for info in output_info]
        
        # 현재는 시간을 측정하는 것이 목적이므로, 실제 복잡한 후처리 로직은 생략하고
        # DtoH 복사 및 간단한 CPU 처리가 포함된다고 가정합니다.
        
        # 후처리 시간도 End-to-End 측정에 포함됨
        pass

    return context, bindings, output_info, norm_type, trt_post_proc

# -------------------------------------------------------------------------
# 3. 메인 엔진
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    # TRT 옵션 추가
    p.add_argument("--model_type", required=True, choices=['rmdetr', 'yolo', 'yolact', 'sparseinst', 'condinst', 'tensorrt'])
    p.add_argument("--project_root", required=True, help="모델 코드가 있는 폴더 경로")
    p.add_argument("--config", default="", help="Config 파일 경로")
    p.add_argument("--config_name", default="", help="YOLACT용 Config 이름")
    p.add_argument("--checkpoint", default="", help=".pth 또는 .pt 파일 경로") # 기본값 ""으로 변경
    p.add_argument("--trt_engine", default="", help="TensorRT .plan 파일 경로") # <--- TRT 인자 추가
    p.add_argument("--img_dir", required=True, help="COCO val2017 경로")
    p.add_argument("--repeat", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda")
    
    # 1. 모델 로드 및 설정
    model, norm_type, post_proc_func = None, 'imagenet', None
    
    if args.model_type == 'rmdetr':
        # model은 PyTorch 모델 인스턴스
        model, norm_type, post_proc_func = run_rmdetr(args, device)
    # ... (다른 PyTorch 모델 로직)
    elif args.model_type == 'tensorrt':
        if not args.trt_engine:
            print("[Error] TensorRT 모델은 --trt_engine 경로를 지정해야 합니다.")
            return
        # model은 TRT Execution Context
        context, bindings, output_info, norm_type, post_proc_func = run_tensorrt_latency(args, device)
        model = context
    else:
        print(f"[Info] {args.model_type}는 아직 구현되지 않았습니다.")
        return

    print(f"\n>>> [Load] Type: {args.model_type.upper()} | Target: {Path(args.checkpoint or args.trt_engine).name}")
    
    # 2. 데이터 로드
    inputs = prepare_real_data(args.img_dir, count=args.repeat, device=device, norm_type=norm_type)

    # 3. 측정
    print(">>> [Measure] Running Inference...")
    stream = torch.cuda.current_stream().cuda_stream # 현재 CUDA 스트림 사용
    
    # Warmup (10회)
    for i in range(10):
        img = inputs[0]
        
        if args.model_type == 'tensorrt':
            # a. HtoD Copy (PyTorch 텐서의 GPU 주소를 TRT 입력 버퍼로 사용)
            # TRT 입력 바인딩 이름이 'input'이라고 가정하고 첫 번째 바인딩에 연결
            cuda.memcpy_dtod_async(bindings[0], img.data_ptr(), img.element_size() * img.numel(), stream)
            
            # b. TRT 실행 (비동기)
            model.execute_async_v2(bindings=bindings, stream_handle=stream)
            
            # c. DtoH Copy / Post-processing
            for idx, info in enumerate(output_info):
                # TRT 출력 버퍼(info['buffer'])에서 데이터 복사 또는 후처리 실행
                pass # End-to-End 측정을 위해 TRT Post-processing 함수 호출
            
            post_proc_func(output_info) # 후처리 함수 호출 (시간 포함)
            
        else:
            # PyTorch 측정 로직
            with torch.no_grad():
                out = model(img)
            if post_proc_func: post_proc_func(out)
            
        torch.cuda.synchronize() # <--- End-to-End 측정의 핵심 동기화

    # 측정 루프 (Repeat 200회)
    t0 = time.perf_counter()
    with torch.no_grad():
        for img in inputs:
            
            if args.model_type == 'tensorrt':
                # a. HtoD Copy
                cuda.memcpy_dtod_async(bindings[0], img.data_ptr(), img.element_size() * img.numel(), stream)
                
                # b. TRT 실행
                model.execute_async_v2(bindings=bindings, stream_handle=stream)
                
                # c. DtoH Copy / Post-processing
                for idx, info in enumerate(output_info):
                    pass # DtoH 또는 GPU Post-processing (시간 포함)

                post_proc_func(output_info) # 후처리 함수 호출
                
            else:
                # PyTorch 측정 로직
                out = model(img)
                if post_proc_func: post_proc_func(out)
            
            # <--- End-to-End 측정을 위한 필수 동기화
            torch.cuda.synchronize() 
            
    t1 = time.perf_counter()

    # 4. 결과 출력
    lat = (t1 - t0) / len(inputs) * 1000
    print(f"--------------------------------------------------")
    print(f"RESULT: {args.model_type.upper()} | {Path(args.checkpoint or args.trt_engine).stem}")
    print(f"Latency: {lat:.2f} ms")
    print(f"FPS    : {1000/lat:.2f}")
    print(f"--------------------------------------------------")

if __name__ == "__main__":
    main()