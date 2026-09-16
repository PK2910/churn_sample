# Same base image as the API, for the same reason: minimal Linux + Python
# 3.10 already installed, "slim" to keep the image small.
FROM python:3.10-slim

WORKDIR /app

# Ingredients before code, same layer-caching reason as api/Dockerfile.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .

EXPOSE 8501

# --server.address=0.0.0.0 makes it accept connections from outside the
# container (Streamlit defaults to localhost-only, which would make it
# unreachable from the outside, same idea as uvicorn's --host 0.0.0.0).
# --server.port uses Render's assigned PORT, falling back to Streamlit's
# usual default (8501) for local runs. --server.headless=true skips
# Streamlit's normal "open a browser automatically" behavior, which doesn't
# make sense inside a container with no browser to open.
CMD streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
