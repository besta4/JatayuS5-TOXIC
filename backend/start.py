"""
System Verification and Startup - Jatayu AFDRN
Autonomous Fraud Detection & Response Network

This script verifies all dependencies and starts the complete Jatayu system:
  - Main FastAPI application (port 8000) with:
    * Real-time transaction monitoring API
    * Batch CSV processing API
    * User authentication & authorization (JWT)
    * Multi-agent fraud detection pipeline
    * Admin, merchant, and customer portals
  - Llama.cpp AI model server (port 8080) for LLM-powered intelligence analysis
  - Redis (optional) for dynamic graph embeddings and real-time pattern detection

Architecture:
  - 5-agent pipeline: Transaction Monitoring → Pattern Detection → Risk Assessment → Alert/Block → Compliance
  - Real-time and batch processing modes
  - Role-based access control (CUSTOMER, MERCHANT, ADMIN)
  - WebSocket streaming for live progress updates
"""
import sys
import subprocess
import importlib
import os
import time
import webbrowser
from pathlib import Path

from config import load_environment

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_missing_packages():
    """Install missing required packages"""
    required = [
        # Core web framework
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("httpx", "httpx"),
        ("websockets", "websockets"),
        ("python-multipart", "multipart"),
        
        # Authentication & security
        ("pyjwt", "jwt"),
        ("passlib", "passlib"),
        ("bcrypt", "bcrypt"),
        
        # AI/ML dependencies
        ("llama-cpp-python", "llama_cpp"),
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("xgboost", "xgboost"),
        ("scikit-learn", "sklearn"),
        ("shap", "shap"),
        
        # Data processing
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        
        # Optional: Redis for dynamic graph features
        ("redis", "redis"),
        
        # Streaming
        ("sse-starlette", "sse_starlette"),
    ]
    
    missing = []
    print("🔍 Checking dependencies...")
    for pip_name, import_name in required:
        if check_package(pip_name, import_name):
            print(f"  ✓ {pip_name}")
        else:
            print(f"  ✗ {pip_name} - MISSING")
            missing.append(pip_name)
    
    if missing:
        print(f"\n📦 Installing {len(missing)} missing packages...")
        for package in missing:
            print(f"  Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print("  ✓ All packages installed")
    else:
        print("  ✓ All dependencies installed")
    
    return len(missing) == 0

def check_model():
    """Check if LLM model is downloaded"""
    model_path = BACKEND_DIR / "models" / "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ LLM Model found ({size_mb:.0f} MB)")
        return True
    else:
        print(f"  ✗ LLM Model not found at {model_path}")
        return False

def check_ml_models():
    """Check if trained ML models exist"""
    data_dir = BACKEND_DIR / "data"
    required_files = [
        "xgb_model_*.pkl",
        "gnn_model_*.pt",
        "embeddings_*.npz",
        "scalers_*.pkl",
        "mappings_*.pkl"
    ]
    
    if not data_dir.exists():
        print("  ⚠️  data/ directory not found")
        return False
    
    found_models = list(data_dir.glob("xgb_model_*.pkl"))
    if found_models:
        print(f"  ✓ XGBoost model found: {found_models[0].name}")
        print(f"  ✓ GNN embeddings: {len(list(data_dir.glob('embeddings_*.npz')))} files")
        return True
    else:
        print("  ⚠️  ML models not found in data/")
        print("     Agent 1 will use fallback scoring")
        return False

def check_ppo_policy():
    """Check if PPO policy exists for Agent 3"""
    load_environment()
    ppo_path = BACKEND_DIR / "risk_ppo_2.pt"
    if ppo_path.exists():
        size_kb = ppo_path.stat().st_size / 1024
        enabled = os.getenv("JATAYU_ENABLE_PPO", "0").strip().lower() in {"1", "true", "yes", "on"}
        mode = "enabled" if enabled else "available, disabled by default"
        print(f"  ✓ PPO policy found ({size_kb:.1f} KB) - {mode}")
        return True
    else:
        print("  ⚠️  PPO policy not found (risk_ppo_2.pt)")
        print("     Agent 3 will use rule-based risk assessment")
        return False

def check_database():
    """Check if database exists"""
    db_path = BACKEND_DIR / "data" / "jatayu.db"
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        print(f"  ✓ Database found ({size_kb:.1f} KB)")
    else:
        print("  ℹ Database will be created on first run")
        print("     Tables: users, accounts, transactions, sessions, etc.")
    return True

def start_server(name, command, port):
    """Start a server in a new window"""
    print(f"\n🚀 Starting {name}...")
    if sys.platform == "win32":
        # Windows: Start in new console window
        process = subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=BACKEND_DIR
        )
    else:
        # Unix: Start in background
        process = subprocess.Popen(command, cwd=BACKEND_DIR)
    
    print(f"  ✓ {name} started (PID: {process.pid})")
    print(f"  📍 Running on port {port}")
    return process

def main():
    load_environment()

    print("=" * 80)
    print("🦅 JATAYU - Autonomous Fraud Detection & Response Network")
    print("=" * 80)
    print()
    print("Multi-Agent AI System for Real-Time Financial Fraud Detection")
    print()
    
    # Step 1: Verify dependencies
    print("[1/7] Verifying dependencies...")
    install_missing_packages()
    print()
    
    # Step 2: Check ML models
    print("[2/7] Checking ML models...")
    check_ml_models()
    check_ppo_policy()
    print()
    
    # Step 3: Check LLM model
    print("[3/7] Checking LLM model...")
    model_exists = check_model()
    if not model_exists:
        print("  ⚠️  LLM not found. Intelligence analysis will be unavailable.")
        print("     Run: python run_llama_server.py (auto-downloads on first run)")
    print()
    
    # Step 4: Check database
    print("[4/7] Checking database...")
    check_database()
    print()

    # Step 5: Check Redis
    print("[5/7] Checking Redis connectivity...")
    redis_available = False
    try:
        import redis as _redis
        redis_url = os.getenv("JATAYU_REDIS_URL", "redis://localhost:6379/0")
        r = _redis.Redis.from_url(
            redis_url,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        r.ping()
        safe_redis_url = redis_url.split("@", 1)[-1] if "@" in redis_url else redis_url
        print(f"  ✓ Redis is running ({safe_redis_url})")
        print("     Dynamic graph embeddings: ENABLED")
        redis_available = True
    except Exception:
        print("  ⚠️  Redis not available")
        print("     Dynamic graph features will be disabled")
        print("     Install: https://redis.io/download")
    print()
    
    # Step 6: Start Llama Server (optional)
    llama_process = None
    if model_exists:
        print("[6/7] Starting LLM Server...")
        llama_cmd = [sys.executable, "run_llama_server.py"]
        llama_process = start_server("LLM Server", llama_cmd, 8080)
        print("  ⏳ Waiting 15 seconds for model initialization...")
        time.sleep(15)
    else:
        print("[6/7] Skipping LLM Server (model not found)")
    print()
    
    # Step 7: Start Main Application
    print("[7/7] Starting Main Application...")
    main_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload", 
                "--host", "0.0.0.0", "--port", "8000"]
    main_process = start_server("Main Application", main_cmd, 8000)
    print("  ⏳ Waiting 5 seconds for startup...")
    time.sleep(5)
    
    # Success!
    print()
    print("=" * 80)
    print("✅ JATAYU IS RUNNING!")
    print("=" * 80)
    print()
    print("🌐 Web Interfaces:")
    print("  🔐 Login:           http://localhost:8000/login")
    print("  👤 Customer Portal: http://localhost:8000/dashboard")
    print("  🏪 Merchant Portal: http://localhost:8000/merchant")
    print("  ⚙️  Admin Console:   http://localhost:8000/admin")
    print("  📊 Batch Upload:    http://localhost:8000/batch")
    print()
    print("📚 Developer Resources:")
    print("  📖 API Docs:        http://localhost:8000/docs")
    print("  🔄 ReDoc:           http://localhost:8000/redoc")
    if llama_process:
        print("  🤖 LLM Server:      http://127.0.0.1:8080")
    print()
    print("🔧 System Components:")
    print("  ✓ 5-Agent Pipeline:")
    print("    1. Transaction Monitoring (XGBoost + GNN)")
    print("    2. Pattern Detection (Mule networks, velocity spikes)")
    print("    3. Risk Assessment (PPO reinforcement learning)")
    print("    4. Alert & Block (Enforcement)")
    print("    5. Compliance Logging (Audit trail)")
    print()
    print("  ✓ Authentication: JWT-based with role-based access control")
    print("  ✓ Real-time API: WebSocket streaming for live updates")
    print("  ✓ Batch Processing: CSV upload with progress tracking")
    if redis_available:
        print("  ✓ Redis: Dynamic graph embeddings enabled")
    else:
        print("  ⚠ Redis: Disabled (static embeddings only)")
    print()
    print("📝 Process Information:")
    if llama_process:
        print(f"  LLM Server PID:  {llama_process.pid}")
    print(f"  Main App PID:    {main_process.pid}")
    print()
    print("🧪 Quick Start:")
    print("  1. Register: POST /auth/register (or use web UI)")
    print("  2. Login: POST /auth/login")
    print("  3. Real-time: POST /transactions/initiate")
    print("  4. Batch: Upload CSV at /batch")
    print()
    print("🛑 To Stop:")
    print("  - Close the console windows, or")
    print("  - Press Ctrl+C in each terminal")
    print()
    print("=" * 80)
    
    # Open browser
    print("\n🌐 Opening login page in browser...")
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8000/login")
    except:
        pass
    
    print("\n✨ System is ready! This window can stay open for monitoring.")
    print("   Both servers are running in separate console windows.")
    print("\nPress Enter to exit this launcher (servers will keep running)...")
    try:
        input()
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Launcher closed. Servers are still running in background.")
    print("   Close their console windows to stop them.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
