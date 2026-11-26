# JobMatch AI
Local prototype with sample job data, TF-IDF cosine ranking, FastAPI REST endpoints, SQLite profile and conversation history, and optional Groq tool calling. No live job feed or production deployment.

1. `python -m venv .venv` then activate the environment.
2. `pip install -r requirements.txt`
3. `python -m pytest -q`
4. `uvicorn app:app --reload` then open http://127.0.0.1:8000/docs.
5. PUT `/profiles/demo` with skills and location, then POST `/chat` with user_id `demo` and question.

Optional live LLM: set GROQ_API_KEY and GROQ_MODEL (a supported tool-calling model). Never commit keys. Optional semantic embeddings: install `sentence-transformers` and set EMBEDDING_MODE=semantic; first run downloads the model. Default mode requires neither. The live LLM and semantic model paths require separate testing; tests use a mocked LLM response and the offline ranker. Dockerfile is provided but image has not been built here. This is a local single-user demo, not an authenticated production service. Sample listings are fictional.
