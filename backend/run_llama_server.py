#!/usr/bin/env python
"""
run_llama_server.py — Helper script to download and run llama.cpp server

This script:
1. Checks/installs required packages (llama-cpp-python, huggingface-hub)
2. Downloads LFM2.5-1.2B-Instruct (Liquid Foundation Model) GGUF
3. Starts the OpenAI-compatible server on port 8080

Usage:
    python run_llama_server.py

The server will expose: http://localhost:8080/v1/chat/completions
which is used by Agent 2 (PatternDetectionAgent) for generating reasoning.

Model: LiquidAI LFM2.5-1.2B-Instruct
- 1.2B parameters, very fast inference
- Optimized for instruction following
- Perfect for fraud pattern reasoning
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Configuration - LiquidAI LFM2.5-1.2B-Instruct (official GGUF)
MODEL_REPO = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
MODEL_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
LOCAL_MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = LOCAL_MODEL_DIR / MODEL_FILE

# Server settings
HOST = "127.0.0.1"
PORT = 8080
# Auto-detect GPU availability
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False

N_GPU_LAYERS = -1 if HAS_GPU else 0  # -1 = all layers to GPU, 0 = CPU only
CONTEXT_SIZE = 4096  # LFM2.5 supports up to 32k context


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=False)


def check_and_install_packages():
    """Ensure required packages are installed."""
    packages_to_check = [
        ("llama_cpp", "llama-cpp-python"),
        ("huggingface_hub", "huggingface-hub"),
    ]
    
    for import_name, pip_name in packages_to_check:
        try:
            __import__(import_name)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            print(f"✗ {pip_name} not found. Installing...")
            
            if pip_name == "llama-cpp-python":
                # Platform-specific installation
                system = platform.system()
                
                if system == "Windows":
                    # Use pre-built wheels for Windows
                    wheel_urls = [
                        # CUDA 12.4 pre-built wheel for Windows (GPU)
                        "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19-cu124/llama_cpp_python-0.3.19-cp312-cp312-win_amd64.whl",
                        # CPU fallback
                        "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19/llama_cpp_python-0.3.19-cp312-cp312-win_amd64.whl",
                    ]
                    
                    installed = False
                    for url in wheel_urls:
                        print(f"  Trying: {url.split('/')[-1]}")
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", url],
                            capture_output=True
                        )
                        if result.returncode == 0:
                            print(f"  ✓ Installed successfully")
                            installed = True
                            break
                        else:
                            print(f"  ✗ Failed, trying next option...")
                    
                    if not installed:
                        print("\n  ERROR: Could not install llama-cpp-python")
                        print("  Please install Visual Studio Build Tools or use pre-built wheel")
                        print("  Manual install: pip install llama-cpp-python --prefer-binary")
                        sys.exit(1)
                else:
                    # Linux/Mac: Use pip with binary preference
                    print(f"  Installing {pip_name} for {system}...")
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", pip_name, "--prefer-binary"],
                        capture_output=False
                    )
                    if result.returncode != 0:
                        print(f"\n  ERROR: Could not install {pip_name}")
                        print(f"  Manual install: pip install {pip_name} --prefer-binary")
                        sys.exit(1)
            else:
                run_cmd([sys.executable, "-m", "pip", "install", pip_name])


def download_model():
    """Download the model if not already present."""
    if MODEL_PATH.exists():
        print(f"✓ Model already exists: {MODEL_PATH}")
        return
    
    print(f"Downloading model: {MODEL_REPO}/{MODEL_FILE}")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    from huggingface_hub import hf_hub_download
    
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    
    if MODEL_PATH.exists():
        print(f"✓ Model downloaded: {MODEL_PATH}")
    else:
        print(f"✗ Download failed. Expected file at: {MODEL_PATH}")
        sys.exit(1)


def start_server():
    """Start the llama.cpp server."""
    print("\n" + "="*60)
    print("Starting llama.cpp server...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Endpoint: http://{HOST}:{PORT}/v1/chat/completions")
    print(f"  GPU Layers: {N_GPU_LAYERS} ({'GPU accelerated' if HAS_GPU else 'CPU only'})")
    print(f"  Hardware: {'CUDA GPU detected' if HAS_GPU else 'CPU mode'}")
    print("="*60 + "\n")
    
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(MODEL_PATH),
        "--host", HOST,
        "--port", str(PORT),
        "--n_gpu_layers", str(N_GPU_LAYERS),
        "--n_ctx", str(CONTEXT_SIZE),
    ]
    
    try:
        # Run server (blocking)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    print("="*60)
    print("Jatayu — llama.cpp Server Setup")
    print("="*60 + "\n")
    
    # Step 1: Check/install packages
    print("[1/3] Checking dependencies...")
    check_and_install_packages()
    print()
    
    # Step 2: Download model
    print("[2/3] Checking model...")
    download_model()
    print()
    
    # Step 3: Start server
    print("[3/3] Starting server...")
    start_server()


if __name__ == "__main__":
    main()
