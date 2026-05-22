export const interventionBody = JSON.stringify({
  farm_location: { lat: 6.2, lon: -2.1 },
  farm_size_ha: 2.5,
  current_yield: 0.8,
  intervention_type: "shade_trees",
  country_code: "GH",
});

export const scenarioBody = JSON.stringify({
  farm_location: { lat: 6.2, lon: -2.1 },
  farm_size_ha: 2.5,
  current_yield: 0.8,
  intervention_type: "shade_trees",
  scenario: "ssp245",
  horizon_year: 2035,
  downscaling_method: "linear_delta",
  country_code: "GH",
});

export function authHeaders(token) {
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}
