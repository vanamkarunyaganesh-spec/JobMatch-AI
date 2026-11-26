# JobMatch AI
FastAPI job matching app with a web page, TF-IDF ranking, SQLite profile and history, and optional Groq tool calling.

Listings are India openings read from selected public Greenhouse and Lever company boards (see `live.py`), cached for one hour. `GET /sources` shows how many jobs each board returned. This is not a complete index of jobs.

## Run
1. `pip install -r requirements.txt`
2. `python -m pytest -q`
3. `uvicorn app:app --reload`, then open http://127.0.0.1:8000

Set `GROQ_API_KEY` and `GROQ_MODEL` for AI answers. Never commit keys. Set `JOBS_SOURCE=sample` to use the fictional data in `jobs.json`.

Demo only: no login, and saved data is not durable on free hosting.
