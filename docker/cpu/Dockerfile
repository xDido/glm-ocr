FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      PIP_NO_CACHE_DIR=1 \
      PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
#   poppler-utils  -> PDF rendering fallback for glmocr
#   libgl1 / libglib2.0-0 -> opencv runtime libs
#   gettext-base   -> envsubst
#   curl           -> healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      poppler-utils \
      libgl1 \
      libglib2.0-0 \
      gettext-base \
      curl \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install glmocr with self-host + server extras, plus gunicorn.
# psutil + pyyaml power the /runtime observability endpoint.
RUN pip install \
      "glmocr[selfhosted,server]" \
      "gunicorn>=21.2,<23" \
      "psutil>=5.9" \
      "pyyaml>=6"

COPY wsgi.py /app/wsgi.py
COPY runtime_app.py /app/runtime_app.py
COPY config.yaml.template /app/config.yaml.template
COPY config.layout-off.template /app/config.layout-off.template
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5002

ENTRYPOINT ["/app/entrypoint.sh"]
