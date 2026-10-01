# Fundus domain shift: a small clinical-style evaluation project

**Question:** a model trained on one fundus dataset looks good internally. What happens to discrimination,
sensitivity/specificity at a fixed operating point, and calibration on images from different cameras and clinics?

This mirrors the core problem for an optician-screening product: images come from many devices, so
internal test scores are not enough.

Task used here: binary **referable diabetic retinopathy** (ICDR grade >= 2) from fundus photos.
DR is a stand-in, because it has large public datasets. The pipeline and the evaluation questions are the
same ones that matter for a rare-disease screening tool.

## Setup
```bash
pip install -r requirements.txt     # on Colab, torch/torchvision are already there
```
Use a GPU (free Colab T4 is fine). On CPU, training will be impractically slow.

## Data (download manually; you need to accept each licence)
- **Train/internal:** APTOS 2019 Blindness Detection (Kaggle), about 3.6k images, grades 0-4.
- **External:** a different dataset graded on the same 0-4 scale, e.g. IDRiD (516 images, India)
  or a Messidor-2 copy. **Check the grading scheme and file format of whichever you use.**
  If the label definitions don't match, the "shift" you measure will be label noise.

## Run
```bash
# 1. Standardise labels (+ stratified 70/15/15 split for the training dataset)
python make_csv.py --images data/aptos/train_images --labels data/aptos/train.csv \
    --id-col id_code --grade-col diagnosis --ext .png --threshold 2 --split --out csv/aptos.csv
python make_csv.py --images data/idrid/images --labels data/idrid/labels.csv \
    --id-col "Image name" --grade-col "Retinopathy grade" --ext .jpg --threshold 2 --out csv/external.csv

# 2. Train (selects on validation AUROC only)
python train.py --csv csv/aptos.csv --out runs/r50

# 3. Predict on val, internal test, external
python predict.py --ckpt runs/r50/best.pt --csv csv/aptos.csv --split val  --out preds/val.csv
python predict.py --ckpt runs/r50/best.pt --csv csv/aptos.csv --split test --out preds/test.csv
python predict.py --ckpt runs/r50/best.pt --csv csv/external.csv           --out preds/external.csv

# 4. Evaluate
python evaluate.py --val preds/val.csv --internal preds/test.csv --external preds/external.csv --out results
```

## What evaluate.py does
1. Fits **temperature scaling** on validation only.
2. Picks an operating threshold on validation that gives 90% sensitivity, then **freezes** it.
3. Reports on the internal test split and the external set, with 95% bootstrap CIs:
   AUROC; sensitivity and specificity at the frozen threshold; specificity at 90% sensitivity when the
   threshold is re-picked on that set; ECE before and after temperature scaling; ROC and reliability plots.

The key comparison is **frozen vs. re-picked threshold on the external set**. A model can keep a decent AUROC
while its fixed operating point silently stops meeting its sensitivity target. That is what a regulator and a
clinical partner care about.

## Results to fill in
| Set | n | AUROC [CI] | Sens @ frozen thr | Spec @ frozen thr | ECE raw -> scaled |
|---|---|---|---|---|---|
| Internal test | | | | | |
| External | | | | | |

## Ideas for one extension (pick one, don't do all)
- Compare a second backbone (e.g. ConvNeXt-T or a ViT) and see whether the shift gap changes.
- Stronger colour/contrast augmentation or Ben Graham-style preprocessing, and its effect on the external gap.
- Simple image-quality gate: flag low-confidence or blurry images and report performance after abstaining.
- Multiple seeds (3-5) to show how much of the gap is just training variance.

## Limitations to state in your write-up
- Single training run unless you add seeds; public datasets are small and not a clinical validation.
- DR is not uveal melanoma; you are demonstrating method, not a clinical claim.
- Label definitions and image populations differ across datasets, so the shift mixes camera, population and labelling effects.
