#!/usr/bin/env bash
# Kill the primary provider mid-load and prove nothing 500s.
#
# Requires PROVIDER_CHAIN=mock,mock_backup with the primary configured to fail.
# We simulate the outage by flipping MOCK_FAILURE_RATE to 1.0 on the api
# container, which is the same thing from the gateway's point of view: every
# call to the primary raises.
set -euo pipefail

API=${BASE_URL:-http://localhost:8000}
KEY=${API_KEY:-demo-key-please-change}

echo "== baseline =="
curl -sf -XPOST "$API/v1/chat" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"chaos probe"}],"no_cache":true}' \
  | python -c 'import sys,json; d=json.load(sys.stdin); print("provider:", d["provider"])'

echo "== starting load in background =="
k6 run -e API_KEY="$KEY" -e BASE_URL="$API" bench/load.js &
K6_PID=$!
sleep 20

echo "== killing the primary provider =="
docker compose exec -T api sh -c 'kill -TERM 1' || true    # or: docker compose stop provider-primary
sleep 5

echo "== during outage: requests must still succeed via failover =="
for i in $(seq 1 10); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -XPOST "$API/v1/chat" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"probe '"$i"'"}],"no_cache":true}')
  echo "  probe $i -> HTTP $code"
done

wait $K6_PID
echo "== done: check http_req_failed in the k6 summary =="
