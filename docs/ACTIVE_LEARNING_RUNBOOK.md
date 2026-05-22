# Active Learning Runbook

Use BSSAL active learning when entering a new cocoa region or when the region has fewer than
200 trusted cocoa/non-cocoa labels. The primary use case is Latin America expansion beyond the
current Ecuador, Peru, and Colombia coverage, where labels are sparse and spatial clustering can
overstate model confidence.

## Workflow

1. Prepare an initial point-label GeoJSON with `label` properties (`0` non-cocoa, `1` cocoa).
2. Run the loop:

   ```bash
   python scripts/run_active_learning_loop.py \
     --region peru \
     --initial-labels data/active/peru/initial_labels.geojson \
     --budget 50 \
     --iterations 5 \
     --pseudo-threshold 0.95
   ```

3. Each iteration writes a reviewer packet to:

   ```text
   data/active/<region>/iter_<n>/to_label.geojson
   ```

4. Iteration reports are written to:

   ```text
   reports/active/<region>/iter_<n>.md
   ```

5. MLflow metrics are logged under experiment `active_learning_<region>`.

## Human-in-the-loop labeling protocol

- Open `to_label.geojson` in QGIS or FieldNotes with the most recent Sentinel-2 composite.
- Label only visible and defensible cocoa/non-cocoa points; leave ambiguous points unlabeled.
- Preserve the point geometry and add one of:
  - `label=1` for cocoa
  - `label=0` for non-cocoa
- Add optional fields `reviewer`, `imagery_date`, and `notes` for auditability.
- Feed the reviewed GeoJSON into the next run as `--initial-labels`.

## Spatial filtering

The BSSAL loop fits 12 monthly NDVI semi-variograms with `scikit-gstat`, takes the minimum fitted
range, projects WGS84 points into the region UTM zone, and discards candidate samples within that
range of an existing label. This follows the Kaijage et al. 2024 spatial decorrelation protocol and
keeps query batches from collapsing into the same plantation cluster.

## Semi-supervised bootstrap

FixMatch-style pseudo-labeling runs every self-training interval. Weak augmentation uses random
crop plus flip; strong augmentation adds RandAugment-style jitter and Sentinel-2 band noise.
Only samples above `PSEUDO_THRESHOLD` (default `0.95`) are merged into the self-training set.

## FTW warm-start guidance

Fields of The World (`ftw-baselines`) weights can warm-start the encoder for non-perennial crop
priors, especially parcel boundary and field texture features. FTW excludes perennial crops, so do
not use FTW outputs as cocoa labels and do not directly transfer FTW crop classes into cocoa/non-cocoa
targets. Use FTW only as an encoder initialization before cocoa-specific supervised or BSSAL
fine-tuning.

## Production exposure backend

After promoting a per-region BSSAL checkpoint to `models/bssal_<region>.pt`, enable the backend with:

```bash
export ACTIVE_LEARNING_ENABLED=true
export COCOA_EXPOSURE_BACKEND=active_learning
```

The default remains disabled to prevent accidental use of experimental checkpoints.
