# Fundus domain shift: a small clinical-style evaluation project

**Question:** a model trained on one fundus dataset looks good internally. What happens to discrimination,
sensitivity/specificity at a fixed operating point, and calibration on images from different cameras and clinics?

This mirrors the core problem for an optician-screening product: images come from many devices, so
internal test scores are not enough.

Task used here: binary **referable diabetic retinopathy** (ICDR grade >= 2) from fundus photos.
DR is a stand-in, because it has large public datasets. The pipeline and the evaluation questions are the
same ones that matter for a rare-disease screening tool.

## Quickest path: Kaggle
Open `kaggle_run.ipynb` on Kaggle (GPU T4, internet on), attach the APTOS 2019 competition data and an IDRiD
mirror that has the *Disease Grading* images + CSVs, set `REPO` to this repo's URL, and run all cells.
It finds the data under `/kaggle/input`, runs every step below and saves `results.zip`.

## Local setup
```bash
pip install -r requirements.txt
```
Use a GPU for training; on CPU it is impractically slow (preprocessing and evaluation are fine on CPU).

## Data (download manually; you need to accept each licence)
- **Train/internal:** APTOS 2019 Blindness Detection (Kaggle), about 3.6k images, grades 0-4 (ICDR).
- **External:** IDRiD Disease Grading (516 images, India, Kowa VX-10a camera), graded on the same ICDR 0-4 scale.
  IDRiD's train and test folders reuse file names (`IDRiD_001.jpg` is in both, as different images), so
  `prepare_idrid.py` flattens them into one folder with unique ids first.
  If you swap in another external set, **check its grading scheme**: if the label definitions don't match,
  the "shift" you measure will be label noise.

## Run
```bash
# 1. Labels: flatten IDRiD; standardise labels; drop duplicate images; stratified 70/15/15 split for APTOS
python prepare_idrid.py --root "data/IDRiD/B. Disease Grading" --out data/idrid_flat
python make_csv.py --images data/aptos/train_images --labels data/aptos/train.csv \
    --id-col id_code --grade-col diagnosis --ext .png --threshold 2 --dedup --split --out csv/aptos.csv
python make_csv.py --images data/idrid_flat/images --labels data/idrid_flat/labels.csv \
    --id-col id --grade-col grade --ext .jpg --threshold 2 --dedup --out csv/external.csv

# 2. Preprocess once: crop borders, pad to square, resize to 512, cache; compute image-quality metrics
#    (APTOS first: the quality normalisation is fitted on its train split)
python preprocess.py --csv csv/aptos.csv    --cache-dir cache/aptos    --stats csv/quality_stats.json --out csv/aptos_cached.csv
python preprocess.py --csv csv/external.csv --cache-dir cache/external --stats csv/quality_stats.json --out csv/external_cached.csv

# 3. Train (selects on validation AUROC only; writes args.json with the config and git hash)
python train.py --csv csv/aptos_cached.csv --out runs/r50

# 4. Predict on val, internal test, external (image size is read from runs/r50/args.json)
python predict.py --ckpt runs/r50/best.pt --csv csv/aptos_cached.csv --split val  --out preds/val.csv
python predict.py --ckpt runs/r50/best.pt --csv csv/aptos_cached.csv --split test --out preds/test.csv
python predict.py --ckpt runs/r50/best.pt --csv csv/external_cached.csv          --out preds/external.csv

# 5. Evaluate
python evaluate.py --val preds/val.csv --internal preds/test.csv --external preds/external.csv --out results
```

## Data hygiene
- **Duplicates:** APTOS 2019 contains the same photo under different ids. Left in, copies land on both sides of
  the split and inflate the internal score (and so the apparent shift gap). `make_csv.py --dedup` removes exact
  perceptual-hash duplicates before splitting and drops duplicate groups whose grades disagree.
- **No stretching:** many fundus images have the circular field cut off top and bottom; images are padded to square
  before resizing so the retina keeps its shape.
- **No patient ids** in APTOS or IDRiD, so the split is per image; both eyes of one patient can still be split apart.

## What evaluate.py does
1. Fits **temperature scaling** on validation only.
2. Picks an operating threshold on validation that gives 90% sensitivity, then **freezes** it.
3. Reports on the internal test split and the external set, with 95% bootstrap CIs:
   AUROC; sensitivity and specificity at the frozen threshold; specificity at 90% sensitivity when the
   threshold is re-picked on that set; ECE before and after temperature scaling; ROC and reliability plots.

4. **Quality gate** (needs the `quality` column from `preprocess.py`): rejects images whose quality score is below the
   10% quantile of validation quality (cutoff frozen, `--abstain-frac`), and reports coverage and metrics on kept vs.
   rejected images per set. `quality_gate.png` sweeps the rejection rate from 0 to 30% and compares against a
   confidence gate (reject images whose logit is closest to the threshold).

The key comparison is **frozen vs. re-picked threshold on the external set**. A model can keep a decent AUROC
while its fixed operating point silently stops meeting its sensitivity target. That is what a regulator and a
clinical partner care about.

## Image-quality gate
At an optician, some photos will be blurred, badly exposed or badly framed. The product question is whether to
grade them anyway or ask for a retake. `preprocess.py` scores each image with four simple metrics
(computed inside the field of view on the 512px cached image):

| Metric | Catches |
|---|---|
| `log_sharpness` = log variance of the Laplacian | blur, defocus |
| `brightness` = mean grey level | under/over-exposure (penalised in both directions) |
| `clip_frac` = share of pixels < 25 or > 240 | dark patches, glare |
| `fov_frac` = field-of-view area / image area | cut-off or badly framed field |

`quality` = mean of the signed z-scores (clipped to +-3), normalised on the APTOS train split only.
Things to look at in the results: does rejecting low-quality images restore sensitivity on the external set,
or does the gate mostly reject *external* images because they come from a different camera? In that case,
coverage drops and the "quality" score is really measuring domain shift.

## Results to fill in
| Set | n | AUROC [CI] | Sens @ frozen thr | Spec @ frozen thr | ECE raw -> scaled |
|---|---|---|---|---|---|
| Internal test | | | | | |
| External | | | | | |

| Quality gate (10% of val rejected) | Coverage | Sens kept / rejected | Spec kept / rejected |
|---|---|---|---|
| Internal test | | | |
| External | | | |

## Further extensions (not done here)
- Compare a second backbone (e.g. ConvNeXt-T or a ViT) and see whether the shift gap changes.
- Stronger colour/contrast augmentation or Ben Graham-style preprocessing, and its effect on the external gap.
- Multiple seeds (3-5) to show how much of the gap is just training variance.
- Grad-CAM on internal vs. external images: is the model looking at lesions or at camera artefacts?
- Validate the quality score against human "gradable / ungradable" labels (e.g. EyeQ on EyePACS).

## Limitations to state in your write-up
- Single training run unless you add seeds; public datasets are small and not a clinical validation.
- IDRiD has 516 images, so external CIs are wide; treat differences within the CIs as noise.
- The quality score is hand-crafted and not validated against human gradability labels.
- DR is not uveal melanoma; you are demonstrating method, not a clinical claim.
- Label definitions and image populations differ across datasets, so the shift mixes camera, population and labelling effects.
