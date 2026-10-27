# Deployment

AmbuRoute is a research prototype with a production-minded service layer. Read [`safety.md`](safety.md) before pointing it at anything real.

## Run it

```bash
# Docker (recommended)
docker build -t ambu-route .
docker run -p 8000:8000 -e MAPBOX_TOKEN=pk.xxx -e API_KEY=choose-a-long-random-key \
  -e STATE_PATH=/data/state.json -e SHADOW_LOG=/data/shadow.jsonl -v ambu-data:/data ambu-route

# or with Compose (reads MAPBOX_TOKEN and API_KEY from your shell or an .env file)
docker compose up --build
```

Open `http://localhost:8000` for the console and `/docs` for the API. Without a Mapbox token the service still answers, using the built-in travel-time model, but the map does not load.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MAPBOX_TOKEN` | none | public `pk.` token for the map and live routes |
| `API_KEY` | none | if set, write endpoints (`/hospitals/*/beds`, `/hospitals/simulate-updates`, `/feedback`, `/decision`) need the `X-API-Key` header; reads and recommendations stay open |
| `CORS_ORIGINS` | none | comma-separated origins allowed to call the API from a browser |
| `MAX_AMBULANCES` | 200 | most ambulances in one `/assign` request |
| `RATE_LIMIT_PER_MIN` | 0 (off) | requests per client address per minute (health checks are exempt) |
| `STALE_MINUTES` | 15 | a hospital bed report older than this is flagged as stale |
| `STATE_PATH` | none | JSON file that keeps bed state and learned trip-time corrections across restarts (written atomically) |
| `SHADOW_LOG` | none | JSONL file of recommendations next to dispatcher decisions (no patient data) |
| `PPO_MODEL` | selected model | path to a PPO checkpoint |
| `LOG_LEVEL` | INFO | JSON access logs with a request id |

## Operating it

- **Health:** `GET /health` (liveness) and `GET /ready` (data loaded, state and log paths writable; also reports how many hospitals have stale bed data). The container has a health check.
- **One worker.** Bed state and learned corrections live in the process. Run a single worker (the default), or move state to a shared store before scaling out; two workers would each see different beds.
- **TLS and authentication.** Put the service behind a reverse proxy that terminates TLS. The API key protects writes only; it is not a user system.
- **Backups.** Copy `STATE_PATH` and `SHADOW_LOG` from the volume. Everything else is in the image or the repository.
- **Upgrades.** Rebuild the image and restart; the state file survives. The state file is ignored if the hospital table changes shape.
- **Routing quota.** `/route` makes one Mapbox call per request. The free quota is about 300 requests a minute; see [`performance.md`](performance.md).
- **Secrets.** Never commit `.env`. The Mapbox token is public by design but should still be restricted to your domain in the Mapbox dashboard.
