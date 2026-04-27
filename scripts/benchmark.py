"""
API latency benchmark for the Diffusion backend.

Profiles the key read endpoints using concurrent requests and reports
p50, p95, p99 latencies and throughput.

Usage:
    python scripts/benchmark.py [--api-url http://localhost:8000] [--concurrency 10] [--requests 200]

Requires: aiohttp (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import time
from dataclasses import dataclass, field

import aiohttp

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

log = logging.getLogger(__name__)


@dataclass
class Result:
    endpoint: str
    latencies: list[float] = field(default_factory=list)
    errors: int = 0

    def add(self, latency_ms: float) -> None:
        self.latencies.append(latency_ms)

    def report(self) -> None:
        n = len(self.latencies)
        total = n + self.errors
        if n == 0:
            print(f"  {self.endpoint:<45}  no successful requests ({self.errors} errors)")
            return
        s = sorted(self.latencies)
        p50 = statistics.median(s)
        p95 = s[int(len(s) * 0.95)]
        p99 = s[int(len(s) * 0.99)]
        mean = statistics.mean(s)
        print(
            f"  {self.endpoint:<45}  "
            f"mean={mean:6.1f}ms  p50={p50:6.1f}ms  p95={p95:6.1f}ms  p99={p99:6.1f}ms  "
            f"ok={n}/{total}"
        )


ENDPOINTS = [
    "/health",
    "/nodes?since_minutes=60&limit=20",
    "/anomalies?limit=20",
    "/anomalies?investigated=false&limit=20",
    "/case-files?limit=20",
    "/case-files?needs_review=false&limit=20",
]


async def hit(
    session: aiohttp.ClientSession,
    api_url: str,
    path: str,
    result: Result,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            async with session.get(
                f"{api_url}{path}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.read()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if resp.status < 500:
                    result.add(elapsed_ms)
                else:
                    result.errors += 1
        except Exception:
            result.errors += 1


async def benchmark_endpoint(
    session: aiohttp.ClientSession,
    api_url: str,
    path: str,
    n_requests: int,
    concurrency: int,
) -> Result:
    result = Result(endpoint=path)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(hit(session, api_url, path, result, semaphore))
        for _ in range(n_requests)
    ]
    await asyncio.gather(*tasks)
    return result


async def run(api_url: str, n_requests: int, concurrency: int) -> None:
    # warm-up
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{api_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    log.error("API returned HTTP %d on /health — is it running?", r.status)
                    return
        except Exception as exc:
            log.error("API unreachable: %s", exc)
            return

        print(f"\nDiffusion API Benchmark")
        print(f"  target      : {api_url}")
        print(f"  requests    : {n_requests} per endpoint")
        print(f"  concurrency : {concurrency}")
        print(f"  endpoints   : {len(ENDPOINTS)}")
        print()

        wall_start = time.perf_counter()

        results: list[Result] = []
        for path in ENDPOINTS:
            result = await benchmark_endpoint(session, api_url, path, n_requests, concurrency)
            results.append(result)

        wall_elapsed = time.perf_counter() - wall_start
        total_requests = sum(len(r.latencies) + r.errors for r in results)
        throughput = total_requests / wall_elapsed

        print(f"{'Endpoint':<47}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  ok/total")
        print("  " + "-" * 100)
        for result in results:
            result.report()

        print()
        print(
            f"  total {total_requests} requests in {wall_elapsed:.1f}s  "
            f"({throughput:.0f} req/s across all endpoints)"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Diffusion API latency benchmark")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--requests", type=int, default=200, help="requests per endpoint")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    await run(args.api_url, args.requests, args.concurrency)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
