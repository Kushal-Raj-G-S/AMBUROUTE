# Performance (measured 2026-10-27)

One `uvicorn` process on a Windows laptop (no Docker), Mapbox token set, 50 concurrent clients, `experiments/load_test.py`. Single measurements on a laptop, not a capacity guarantee.

| Endpoint | Requests | Errors | Throughput | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| `/assign` (8 ambulances, equilibrium) | 600 | 0 | 225 / s | 168 ms | 605 ms | 945 ms |
| `/assign` clinical (8 ambulances, severity + specialty) | 600 | 0 | 225 / s | 162 ms | 560 ms | 811 ms |
| `/route` (decision + live Mapbox route) | 60 | 0 | 14 / s | 3.0 s | 4.0 s | 4.1 s |

- The decision itself is cheap (about 0.3 ms per ambulance in the simulator; the `/assign` latencies above are mostly queueing at 50 concurrent clients on one process).
- `/route` is bounded by the routing provider call. A single request takes about 0.6 s end to end; at 50 concurrent clients latency grows to several seconds. Mapbox's free Directions quota is about 300 requests per minute, so a real deployment needs a paid plan, a route cache for repeated origin/destination pairs, or more workers. The built-in fallback (OSRM, then the calibrated model) keeps responses available if the provider fails.
- Not yet measured: a container (Docker was not running on the test machine; CI builds and starts the image), multiple workers, and a database-backed state store.

Reproduce:

```bash
python -m uvicorn amburoute.api:app --port 8012
python experiments/load_test.py http://127.0.0.1:8012 --requests 600 --concurrency 50 --routes 60
```
