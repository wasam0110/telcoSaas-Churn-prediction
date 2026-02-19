````markdown
Deployment guide — Local run and Streamlit Community Cloud

Overview

- Dashboard: Streamlit Community Cloud (free for public GitHub repos)

Prerequisites

- Push your repository to GitHub (public for Streamlit Cloud)
- Ensure `models/` contains the serialized artifacts the API/dashboard expect
- `requirements.txt` must list all Python dependencies (already present)

Local testing

1. Start the API locally (FastAPI + Uvicorn):

```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

2. Start the dashboard locally (Streamlit):

```bash
python -m streamlit run dashboard/app.py --server.port 8501 --server.address 127.0.0.1
```

3. Visit local endpoints:

- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs

Streamlit Community Cloud (dashboard)

1. Make sure your repo is public on GitHub.
2. Go to https://share.streamlit.io and log in with GitHub.
3. "New app" → choose your repo → set the main file path to `dashboard/app.py` → deploy.
4. Optional: set secrets/environment variables in the Streamlit dashboard settings.

Notes:

- Streamlit Cloud reads your repo and runs `streamlit run dashboard/app.py` automatically. No Dockerfile required for Streamlit Cloud.
- If `models/` contains large binary files, use a remote model store (S3, Git LFS) to keep repo size reasonable.

Post-deploy checks (if using Streamlit Cloud)

- Dashboard: visit the Streamlit URL provided by Streamlit Cloud

If you'd like, I can:

- Add concise local-run instructions to `README.md` (optional)
````
