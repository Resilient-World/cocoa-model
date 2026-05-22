import http from "k6/http";
import { check, sleep } from "k6";
import { scenarioBody, authHeaders } from "./payloads.js";

const baseUrl = __ENV.URL || "http://localhost:8000";
const token = __ENV.TOKEN || "";
const smoke = __ENV.K6_SMOKE === "1";

export const options = smoke
  ? {
      vus: 10,
      duration: "2m",
      thresholds: {
        http_req_failed: ["rate<0.01"],
        http_req_duration: ["p(95)<1500"],
      },
    }
  : {
      scenarios: {
        constant_rps: {
          executor: "constant-arrival-rate",
          rate: 50,
          timeUnit: "1s",
          duration: "10m",
          preAllocatedVUs: 50,
          maxVUs: 100,
        },
      },
      thresholds: {
        http_req_failed: ["rate<0.01"],
        http_req_duration: ["p(95)<1500"],
      },
    };

export default function () {
  const res = http.post(
    `${baseUrl}/simulate-scenario`,
    scenarioBody,
    { headers: { "Content-Type": "application/json", ...authHeaders(token) } }
  );
  check(res, { "status is 200 or 400": (r) => r.status === 200 || r.status === 400 });
  sleep(0.1);
}
