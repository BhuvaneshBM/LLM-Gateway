// k6 load test. ALWAYS runs against PROVIDER_CHAIN=mock (DESIGN.md §5c):
// benchmarking a real provider measures THEIR latency variance, not ours.
//
// Local:
//   k6 run -e API_KEY=demo-key-please-change bench/load.js
//
// Docker:
//   docker run --rm --network llm-gateway_default \
//     -v "$(pwd):/work" -w /work grafana/k6 \
//     run -e BASE_URL=http://api:8000 \
//     -e API_KEY=demo-key-please-change bench/load.js

import http from "k6/http";
import { check } from "k6";
import { Rate, Trend } from "k6/metrics";

const cacheHits = new Rate("cache_hit_rate");
const hitLatency = new Trend("latency_cache_hit_ms");
const missLatency = new Trend("latency_cache_miss_ms");

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY;

const POOL = [
  "What is the capital of France?",
  "What's the capital city of France?",
  "How do I reset my password?",
  "How can I reset my password?",
  "Summarise this document",
  "Give me a summary of this document",
  "When are you open?",
  "What are your business hours?",
];

export const options = {
  scenarios: {
    warm: {
      executor: "ramping-arrival-rate",
      startRate: 10,
      timeUnit: "1s",
      preAllocatedVUs: 100,
      maxVUs: 400,
      stages: [
        { target: 50, duration: "30s" },
        { target: 200, duration: "1m" },
        { target: 200, duration: "2m" },
        { target: 0, duration: "30s" },
      ],
    },
  },

  thresholds: {
    http_req_failed: ["rate<0.01"],
    latency_cache_hit_ms: ["p(95)<50"],
    http_req_duration: ["p(95)<2500"],
  },
};

export default function () {
  const body = JSON.stringify({
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant.",
      },
      {
        role: "user",
        content: POOL[Math.floor(Math.random() * POOL.length)],
      },
    ],
    temperature: 0.0,
    max_tokens: 128,
  });

  const res = http.post(`${BASE_URL}/v1/chat`, body, {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  if (res.status === 200) {
    const data = res.json();

    cacheHits.add(data.cached);
    (data.cached ? hitLatency : missLatency).add(data.latency_ms);
  }
}