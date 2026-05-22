import http from "k6/http";
import { check, sleep } from "k6";
import { interventionBody, authHeaders } from "./payloads.js";

const baseUrl = __ENV.URL || "http://localhost:8000";
const token = __ENV.TOKEN || "";
const smoke = __ENV.K6_SMOKE === "1";

export const options = smoke
  ? {
      vus: 5,
      duration: "1m",
      thresholds: {
        http_req_failed: ["rate<0.01"],
        http_req_duration: ["p(95)<500"],
      },
    }
  : {
      stages: [
        { duration: "2m", target: 20 },
        { duration: "5m", target: 50 },
        { duration: "2m", target: 0 },
      ],
      thresholds: {
        http_req_failed: ["rate<0.005"],
        http_req_duration: ["p(95)<500"],
      },
    };

export default function () {
  const res = http.post(
    `${baseUrl}/simulate-intervention`,
    interventionBody,
    { headers: { "Content-Type": "application/json", ...authHeaders(token) } }
  );
  check(res, { "status 200": (r) => r.status === 200 });
  sleep(0.2);
}
