FROM python:3.12-slim

WORKDIR /app
# CPU-only PyTorch keeps the image small; the rest comes from requirements.txt
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY amburoute ./amburoute
COPY data ./data
COPY results ./results

# Run as an unprivileged user; /data holds the optional state file and shadow log (mount a volume there)
RUN useradd --create-home --uid 10001 ambu && mkdir /data && chown ambu /data
USER ambu

# Pass your Mapbox token at run time:  docker run -p 8000:8000 -e MAPBOX_TOKEN=pk... ambu-route
# Optional: -e API_KEY=... -e STATE_PATH=/data/state.json -e SHADOW_LOG=/data/shadow.jsonl -v ambu-data:/data
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "uvicorn", "amburoute.api:app", "--host", "0.0.0.0", "--port", "8000"]
