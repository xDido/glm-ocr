// k6 scenario for the GLM-OCR CPU container.
//
// Run:
//   k6 run ocr_load.js -e HOST=http://localhost:5002 \
//          --summary-export=../results/k6.json
//
// Tweak VUs / duration via the stages below or `-e VUS=…`.

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';

const HOST = __ENV.HOST || 'http://localhost:5002';

if (!__ENV.IMAGES) {
  throw new Error(
    'ocr_load.js: IMAGES env var is required. Run via scripts/omnidoc_k6.sh, ' +
    'or set IMAGES=<comma-separated-urls> yourself.'
  );
}
const IMAGES = __ENV.IMAGES.split(',').map(s => s.trim()).filter(Boolean);

// When DURATION is set (e.g. by the omnidoc orchestrator) swap the default
// ramp for a constant-VU hold so every driver runs for the same wall-clock
// budget. Otherwise keep the three-stage ramp for manual `make load-k6`.
const DURATION = Number(__ENV.DURATION || 0);
const VUS_HIGH = Number(__ENV.VUS_HIGH || 48);

const scenarios = DURATION > 0
  ? {
      constant: {
        executor: 'ramping-vus',
        startVUs: 1,
        stages: [
          { duration: '10s', target: VUS_HIGH },
          { duration: `${Math.max(DURATION - 20, 1)}s`, target: VUS_HIGH },
          { duration: '10s', target: 0 },
        ],
        gracefulRampDown: '15s',
      },
    }
  : {
      ramp: {
        executor: 'ramping-vus',
        startVUs: 1,
        stages: [
          { duration: '30s', target: Number(__ENV.VUS_LOW || 8) },
          { duration: '1m',  target: Number(__ENV.VUS_MID || 24) },
          { duration: '1m',  target: VUS_HIGH },
          { duration: '30s', target: 0 },
        ],
        gracefulRampDown: '15s',
      },
    };

export const options = {
  discardResponseBodies: false,
  thresholds: {
    'http_req_failed':                 ['rate<0.02'],
    'http_req_duration{status:200}':   ['p(95)<15000', 'p(99)<30000'],
    'checks':                          ['rate>0.98'],
  },
  scenarios,
};

const latency = new Trend('ocr_latency_ms', true);
const fails   = new Counter('ocr_failures');

function pickImage() { return IMAGES[Math.floor(Math.random() * IMAGES.length)]; }

export default function () {
  const body = JSON.stringify({ images: [pickImage()] });
  const res = http.post(`${HOST}/glmocr/parse`, body, {
    headers: { 'Content-Type': 'application/json' },
    timeout: '300s',
    tags:    { endpoint: 'parse' },
  });

  latency.add(res.timings.duration);
  const ok = check(res, { 'status 200': (r) => r.status === 200 });
  if (!ok) fails.add(1);

  sleep(0.1 + Math.random() * 0.4);
}
