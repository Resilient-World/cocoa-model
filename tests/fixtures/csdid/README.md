# CS-DID replication fixtures

`mpdta.csv` is a 500-county × 2004–2007 panel mimicking the structure of `did::mpdta`
(Callaway & Sant'Anna 2021 minimum-wage application).

`mpdta_benchmarks.json` pins `att_gt(g,t)` and `simple_att` from
`scripts/generate_mpdta_fixture.py` (DR + multiplier bootstrap, `n_boot=199`).

Regenerate after estimator changes:

```bash
python scripts/generate_mpdta_fixture.py
```

For exact R `did` package parity, export `mpdta` from R and replace benchmarks with
`att_gt()` output; tests assert ±0.01 against the committed JSON.
