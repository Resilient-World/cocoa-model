import http from "k6/http";
import { check, sleep } from "k6";
import { interventionBody, scenarioBody, authHeaders } from "./payloads.js";

const baseUrl = __ENV.URL || "http://localhost:8000";
const token = __ENV.TOKEN || "";
const smoke = __ENV.K6_SMOKE === "1";

export const options = smoke
  ? { vus: 8, duration: "2m" }
  : {
      vus: 30,
      duration: "5m",
      thresholds: { http_req_failed: ["rate<0.02"] },
    };

export default function () {
  const headers = {
    "Content-Type": "application/json",
    ...authHeaders(token),
  };
  const pick = __ITER % 3;
  let res;
  if (pick === 0) {
    res = http.post(`${baseUrl}/simulate-intervention`, interventionBody, { headers });
    check(res, { "intervention ok": (r) => r.status === 200 });
  } else if (pick === 1) {
    res = http.post(`${baseUrl}/simulate-scenario`, scenarioBody, { headers });
    check(res, { "scenario ok": (r) => r.status === 200 || r.status === 400 });
  } else {
    res = http.get(`${baseUrl}/health`, { headers });
    check(res, { "health ok": (r) => r.status === 200 });
  }
  sleep(0.15);
}
