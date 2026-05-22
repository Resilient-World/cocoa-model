# Aurora vs NeuralGCM vs CorrDiff (2026-05-22)

Aurora 1.5 (Bodnar et al., Nature 2025) compared to NeuralGCM stub and CorrDiff/linear placeholders at 10-day lead across eight cocoa-belt regions.

| Region | Backend | Variable | RMSE | Anomaly corr | CRPS |
|--------|---------|----------|------|--------------|------|
| cameroon | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| cameroon | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| cameroon | aurora | wind10m | 0.351 | nan | 0.298 |
| cameroon | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| cameroon | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| cameroon | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| cameroon | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| cameroon | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| cameroon | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| cameroon | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| cameroon | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| cameroon | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| civ | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| civ | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| civ | aurora | wind10m | 0.351 | nan | 0.298 |
| civ | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| civ | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| civ | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| civ | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| civ | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| civ | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| civ | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| civ | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| civ | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| colombia | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| colombia | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| colombia | aurora | wind10m | 0.351 | nan | 0.298 |
| colombia | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| colombia | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| colombia | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| colombia | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| colombia | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| colombia | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| colombia | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| colombia | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| colombia | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| ecuador | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| ecuador | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| ecuador | aurora | wind10m | 0.351 | nan | 0.298 |
| ecuador | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| ecuador | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| ecuador | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| ecuador | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| ecuador | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| ecuador | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| ecuador | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| ecuador | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| ecuador | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| ghana | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| ghana | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| ghana | aurora | wind10m | 0.351 | nan | 0.298 |
| ghana | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| ghana | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| ghana | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| ghana | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| ghana | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| ghana | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| ghana | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| ghana | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| ghana | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| indonesia | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| indonesia | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| indonesia | aurora | wind10m | 0.351 | nan | 0.298 |
| indonesia | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| indonesia | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| indonesia | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| indonesia | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| indonesia | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| indonesia | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| indonesia | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| indonesia | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| indonesia | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| nigeria | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| nigeria | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| nigeria | aurora | wind10m | 0.351 | nan | 0.298 |
| nigeria | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| nigeria | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| nigeria | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| nigeria | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| nigeria | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| nigeria | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| nigeria | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| nigeria | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| nigeria | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| peru | aurora | 2m_temperature | 0.775 | -0.245 | 0.608 |
| peru | aurora | precipitation | 2.390 | -0.017 | 2.183 |
| peru | aurora | wind10m | 0.351 | nan | 0.298 |
| peru | aurora | surface_solar_radiation | 1.392 | nan | 1.110 |
| peru | neuralgcm_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| peru | neuralgcm_stub | precipitation | 2.390 | -0.017 | 2.183 |
| peru | neuralgcm_stub | wind10m | 0.351 | nan | 0.298 |
| peru | neuralgcm_stub | surface_solar_radiation | 1.392 | nan | 1.110 |
| peru | corrdiff_stub | 2m_temperature | 0.775 | -0.245 | 0.608 |
| peru | corrdiff_stub | precipitation | 2.390 | -0.017 | 2.183 |
| peru | corrdiff_stub | wind10m | 0.351 | nan | 0.298 |
| peru | corrdiff_stub | surface_solar_radiation | 1.392 | nan | 1.110 |

## Limitations

- Aurora provides no strict performance guarantees; biases inherit from ERA5/ERA training.
- Commercial deployment requires `AURORA_COMMERCIAL_OK` and Microsoft approval (AIWeatherClimate@microsoft.com).
- Full GPU backtest with held-out ERA5 Zarr replaces stub truth when `--era5-zarr` is set.
