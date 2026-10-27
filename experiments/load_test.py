"""Small load test for a running service (no extra dependencies).

    python experiments/load_test.py http://127.0.0.1:8000 --requests 600 --concurrency 50

Measures end-to-end latency and throughput of /assign (our decision only, no external routing call) and of /route
(decision plus the routing provider call, so it is bounded by Mapbox latency), and prints one JSON line each.
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx

AMB = {"lat": 12.97, "lon": 77.59}
CASES = {
    "/assign": {"ambulances": [AMB] * 8},
    "/assign (clinical)": {"ambulances": [AMB] * 8, "severity": [2, 1, 0, 2, 1, 0, 2, 1],
                           "specialty": ["cardiac", "stroke", "none", "trauma", "none", "none", "cardiac", "none"]},
    "/route": {"ambulance": AMB, "needs_icu": False},
}


async def run(base, path, body, n, conc):
    sem = asyncio.Semaphore(conc)
    lat, errors = [], 0
    async with httpx.AsyncClient(base_url=base, timeout=30) as c:
        async def one():
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                try:
                    r = await c.post(path.split(" ")[0], json=body)
                    errors += r.status_code != 200
                except httpx.HTTPError:
                    errors += 1
                lat.append((time.perf_counter() - t) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(n)])
        wall = time.perf_counter() - t0
    q = statistics.quantiles(lat, n=100)
    return dict(endpoint=path, requests=n, concurrency=conc, errors=errors, rps=round(n / wall, 1),
                p50_ms=round(statistics.median(lat), 1), p95_ms=round(q[94], 1), p99_ms=round(q[98], 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--routes", type=int, default=100, help="requests for /route (each calls the routing provider)")
    a = ap.parse_args()
    for path, body in CASES.items():
        print(json.dumps(asyncio.run(run(a.base, path, body, a.routes if path == "/route" else a.requests, a.concurrency))), flush=True)
