# The base image: a minimal Linux system with Python 3.10 already installed.
# "slim" keeps the image smaller than the default Python image.
FROM python:3.10-slim

# Everything from here on happens inside /app inside the container.
WORKDIR /app

# Copied and installed separately from the rest of the code on purpose —
# Docker caches each instruction as a "layer". As long as requirements.txt
# doesn't change, Docker reuses this layer on future builds instead of
# reinstalling every package from scratch, which makes rebuilds much faster
# when you're only editing app.py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code, the saved model files, and the
# preprocessing module — the pickled pipeline references these functions by
# module path, so this file needs to exist here for joblib to load it.
COPY app.py .
COPY preprocessing.py .
COPY xgb_pipeline.joblib .
COPY xgb_best_model.joblib .

# Documents that the container listens on port 8000 by default (informational
# — doesn't actually publish the port by itself, that happens with
# `docker run -p` locally, or automatically on a platform like Render).
EXPOSE 8000

# Shell form (not the ["..."] array form) on purpose — array-form CMD doesn't
# expand environment variables, so ${PORT:-8000} would be passed through
# literally instead of being substituted. This uses whatever port a cloud
# platform assigns via the PORT environment variable, falling back to 8000
# for local `docker run` where no PORT is set.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
