# Jatayu AFDRN

Jatayu (Autonomous Fraud Detection and Response Network) is a real-time fraud detection platform built with FastAPI, SQLite, and a sequential multi-agent AI pipeline.

It supports both:

- Real-time transaction processing with role-based portals (Customer, Merchant, Admin, Support)
- Batch CSV analysis with WebSocket progress streaming and audit outputs

## Core Capabilities

- 5-stage fraud decision pipeline:
  - Agent 1: Transaction monitoring (XGBoost + GNN artifact path, stub fallback)
  - Agent 2: Pattern detection (mule network, account takeover, velocity spikes)
  - Agent 3: Risk assessment (PPO policy with rule-based fallback)
  - Agent 4: Alert and action decision (PASS, FLAG, HOLD, BLOCK)
  - Agent 5: Compliance logging (structured audit entries)
- JWT auth with RBAC for CUSTOMER, MERCHANT, and ADMIN flows
- SQLite-backed persistence for users, sessions, accounts, transactions, reports, support tickets, and batch tasks
- Frontend pages served directly by FastAPI under static/
- Optional local LLM intelligence generation via llama.cpp server on port 8080

## Tech Stack

- Python 3.12+
- FastAPI + Uvicorn
- SQLite
- pandas, numpy, scikit-learn, xgboost, torch, torch-geometric
- Optional local LLM: llama-cpp-python + huggingface-hub

## Project Layout

```text
jatayu/
  main.py                      # FastAPI app + batch API + static routes
  database.py                  # SQLite layer and schema initialization
  start.py                     # Startup helper (dependency checks + server launch)
  run_llama_server.py          # Local llama.cpp server helper
  risk_ppo_2.pt                # PPO model checkpoint (small)
  mule_test.csv                # Sample batch input
  agents/                      # Multi-agent fraud pipeline
  auth/                        # JWT, password, dependencies, RBAC models
  routers/                     # Auth/users/transactions/admin/merchant/support APIs
  static/                      # HTML/CSS/JS frontend
  models/                      # Local GGUF model directory (ignored in git)
  data/                        # ML artifacts directory (ignored in git)
```

## Prerequisites

- Python 3.12 or newer
- pip
- (Optional) CUDA-capable GPU for faster local LLM inference

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Install core API/runtime packages if your environment does not already include them.

```powershell
pip install fastapi uvicorn python-multipart sse-starlette PyJWT
```

4. Create local environment variables.

- Copy .env.example to .env and set a strong secret.
- Important: this project currently reads environment variables from the OS process, not by auto-loading .env.

PowerShell example:

```powershell
$env:JATAYU_JWT_SECRET = "replace-with-a-strong-random-secret"
$env:JATAYU_AUDIT_FILE = "audit.jsonl"
```

## Run the Project

### Option A: One-command startup helper

```powershell
python start.py
```

This helper checks dependencies, verifies model presence, starts the local LLM server, then starts the FastAPI app.

### Option B: Manual startup (recommended for debugging)

Terminal 1:

```powershell
python run_llama_server.py
```

Terminal 2:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Access Points

- App root: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Local LLM server: http://127.0.0.1:8080

Main UI routes:

- /login
- /dashboard
- /merchant
- /admin
- /batch
- /suspended

## API Surface (High Level)

- /auth
  - register, login, refresh, logout, me, password update
- /users
  - profile, settings, accounts, payees
- /transactions
  - create transaction, verify otp, list/history, details, timeline
- /admin
  - user management, fraud review, compliance reporting, system metrics
- /merchant
  - payments, analytics, settlements
- /support
  - ticketing and admin ticket resolution
- /batch-api
  - upload, websocket progress, results, summary, audit, tasks, intelligence

Use /docs for the exact request and response schemas.

## Model and Artifact Notes

- models/\*.gguf is intentionally excluded from git due to size.
- data/ artifacts are excluded from git; if missing, Agent 1 falls back to stub scoring mode.
- risk_ppo_2.pt is included and used by the risk agent when present.
- If the local LLM server is not available, the rest of the fraud pipeline still runs.

## Security Notes

- Replace JATAYU_JWT_SECRET before running outside local development.
- Do not commit .env, database files, or runtime audit logs.
- JWT secret changes invalidate previously issued tokens.

## Common Issues

- Missing FastAPI/Uvicorn imports:
  - Install runtime packages from the Setup section.
- llama-cpp installation/download issues:
  - Run without LLM first; the fraud pipeline still functions.
  - Ensure model file exists at models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf if using local intelligence streaming.
- Empty batch results:
  - Validate CSV headers against expected transaction fields used by main.py.

## License

No license file is currently present in this repository. Add one if you plan to distribute publicly.
