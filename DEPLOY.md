# Deploy RecovAI on Vercel

Single FastAPI app (`main.py`) serves the UI at `/` and all API routes on the same origin.

## Prerequisites

- [Vercel account](https://vercel.com) and [Vercel CLI](https://vercel.com/docs/cli) ≥ 48.1.8 (optional; Git integration works too)
- [Groq API key](https://console.groq.com)
- **Postgres** for prediction history (do **not** use SQLite on Vercel—the filesystem is ephemeral)

**Postgres on Vercel:** Vercel no longer runs its own Postgres product. In the dashboard, **Storage → Create Database → Postgres** installs the **Neon** integration. The **Neon Free** plan ($0/month, no card required on Neon directly) includes about **1 GB storage** and **100 CU-hours** per project per month—enough for RecovAI history. Hobby Vercel teams use that free tier through the Marketplace; usage above free limits is billed via Vercel/Neon.

## 1. Create the database

1. Create a Neon project and database.
2. Copy the connection string (often starts with `postgres://`).
3. In Vercel → Project → **Settings → Environment Variables**, add:

| Variable | Environments | Notes |
|----------|--------------|--------|
| `DATABASE_URL` | Production, Preview | Neon connection string |
| `GROQ_API_KEY` | Production, Preview | Server-side only; never commit |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | Production, Preview | Set to `1` if the build fails on bundle size (ML deps) |

Tables are created automatically on first request (`init_db()` at startup).

## 2. Deploy

**Git (recommended)**

1. Import the GitHub repo in Vercel (or use the linked `recovai-backend` project).
2. Root directory: repository root.
3. Framework preset: Vercel should detect **FastAPI** from `main.py`.
4. Deploy.

**CLI**

```bash
cd HCL-MCP-AI-project-
vercel link    # if not already linked
vercel env pull .env.local   # optional, for local vercel dev
vercel --prod
```

## 3. Verify

| Check | URL |
|-------|-----|
| UI | `https://<your-project>.vercel.app/` |
| Health | `https://<your-project>.vercel.app/health` |
| Readiness | `https://<your-project>.vercel.app/api/test` |
| Swagger | `https://<your-project>.vercel.app/docs` |

The first request after idle may take longer while models load (cold start). A cron hits `/health` every 6 hours to reduce long idle periods.

## Local dev with Vercel

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
vercel dev
```

Use `DATABASE_URL` pointing at Neon (or omit for local SQLite).

## Bundle size

Python functions have a **500 MB** uncompressed limit. XGBoost, scikit-learn, SHAP, and matplotlib are heavy. If deploy fails with a size error:

1. Set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in the project and redeploy.
2. Trim unused training scripts (already excluded in `vercel.json` where possible).

## Container fallback

If the managed Python build fails, use the Docker image defined in `deploy/Dockerfile.vercel`:

1. Copy it to the repo root as `Dockerfile.vercel` (Vercel auto-detects only that filename at root).
2. Redeploy.

Remove `Dockerfile.vercel` from the root when switching back to native Python.

## What not to use on Vercel

- **SQLite** for production history (`sqlite:///./recovai.db`) — data will not persist reliably.
- **Render `DATABASE_URL`** unless you keep that database running elsewhere.
