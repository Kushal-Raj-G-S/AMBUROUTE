FROM python:3.12-slim

WORKDIR /app
# CPU-only PyTorch keeps the image small; the rest comes from requirements.txt
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY amburoute ./amburoute
COPY data ./data
COPY results ./results

# Pass your Mapbox token at run time:  docker run -p 8000:8000 -e MAPBOX_TOKEN=pk... ambu-route
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "amburoute.api:app", "--host", "0.0.0.0", "--port", "8000"]
