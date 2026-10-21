<div align="center">
<h1 align="center">TransientTrack</h1>
<h2 align="center">Advanced Multi-Object Tracking and Classification of Cancer Cells with Transient Fluorescent Signals</h2>

<a href="https://doi.org/10.1109/TMI.2026.3744441">
  <img src="https://img.shields.io/badge/Paper-IEEE TMI-green?style=flat&label=Paper&link=https://doi.org/10.1109/TMI.2026.3744441" alt="Static Badge">
</a>
<a href="https://repository.publisso.de/resource/frl:6526608">
  <img src="https://img.shields.io/badge/Dataset-blue?style=flat&label=Data&link=https://repository.publisso.de/resource/frl:6526608"
  alt="Static Badge" />
</a>
<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Static Badge" />
</a>
</div>

Official implementation of **TransientTrack: Advanced Multi-Object Tracking and Classification of Cancer Cells with Transient Fluorescent Signals**, accepted for publication in *IEEE Transactions on Medical Imaging* (2026).

<p align="center">
  <img alt="Overview Image" src="./assets/overview.svg">
  <br>
    <em>Overview</em>
</p>

TransientTrack is a deep learning framework for tracking cells in time-lapse microscopy. Alongside detection and tracking, it distinguishes living, dividing, and dead cells, so mitosis and apoptosis are recovered as part of tracking rather than as a separate step, and complete trajectories with lineage information follow directly.

Association is deliberately lightweight: matching runs on the decoder feature embeddings of the detection transformer together with spatial information, with no re-identification model, contrastive objective, or hand-crafted appearance features. Multi-stage matching over all detection boxes and Kalman-filter interpolation of missing tracklets keep trajectories intact when detections are intermittent.

The method was developed for a case that breaks trackers assuming a temporally stable appearance: multi-channel recordings whose fluorescent signals fluctuate over time, following processes such as the circadian rhythm.

This repository contains the training and inference code, the evaluation suite (detection, tracking, CTC, and event metrics), and the downstream lineage analysis behind the single-cell cisplatin dose-response study in the paper.

---

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
  - [Dataset](#1-dataset)
  - [CTC Evaluation Software](#2-ctc-evaluation-software)
  - [Ablation Baselines](#3-ablation-baselines-optional)
  - [Model Weights](#4-model-weights)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Analysis](#analysis)
- [Using TransientTrack on Your Own Data](#using-transienttrack-on-your-own-data)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Citation](#citation)
- [License](#license)

---

## Installation

The model weights are stored with [Git LFS](https://git-lfs.com), so install and initialise it before cloning. 

```bash
git lfs install

git clone https://github.com/bozeklab/TransientTrack.git
cd TransientTrack
conda env create -f environment.yml
conda activate ttrack
```

---

## Setup

The dataset and the CTC evaluation binaries are **not** included in this repository and must be downloaded before training or evaluation. The ablation baselines are optional and only needed to re-run the baseline rows.

### 1. Dataset

The dataset released with the paper contains **309 multi-channel fluorescence microscopy videos** of human osteosarcoma-derived U-2 OS cells: 234 frames of 1024x1024 pixels per video, captured at 30-minute intervals over 117 hours. Three fluorescent reporters are reported per video — hGeminin-CFP (cell cycle), NR1D1::VNP (circadian clock), and p53-mKate (cell fitness) — all of which vary in intensity over time with the underlying biological process. Cisplatin was added after 48 hours (frame 96) in 118 videos. 275 videos carry manual annotations of cell positions, mitosis and apoptosis events, and lineage relationships, covering frames 49–234.

Download it from [here](https://doi.org/10.4126/FRL01-006526608) and extract it into `src/data/dataset/`:

```
src/data/dataset/
├── annotations.json    # CocoVid-style annotations (videos, images, tracks)
├── 001/
│   ├── images          # Folder containing the .tif files in the format 001.tif
│   ├── ctc_files/      # CTC-format ground truth for this position
│   └── motm_001.csv    # Ground truth in MOTM format
├── 002/
│   ├── images/
│   ├── ctc_files/
│   └── motm_002.csv 
└── ...
```

Each video is a directory named by its **position ID** (`001`, `002`, …), matching the `name` field of the corresponding entry in `annotations.json`.


### 2. CTC Evaluation Software

The DET / TRA / LNK scores are computed by the official [Cell Tracking Challenge](http://celltrackingchallenge.net/evaluation-methodology/) evaluation binaries. Download the software for your platform and place it so that the executables sit directly inside `src/utils/ctc/CTC_eval/`:

```
src/utils/ctc/CTC_eval/
├── <platform>          # "Linux", "Windows" or "Mac"
│   ├── DETMeasure  
│   └── TRAMeasure
└── ...
```

### 3. Ablation Baselines (optional)

To run the SORT and DeepSORT ablations, fetch the upstream implementations into the directories below. This repository ships only adapters that call them. Both are GPL-3.0 and therefore not redistributed here; ByteTrack is MIT and is included. The upstream sources are used unmodified.

```bash
# SORT
wget -O src/ablation/SORT/sort.py \
  https://raw.githubusercontent.com/abewley/sort/master/sort.py

# DeepSORT
git clone --depth 1 https://github.com/nwojke/deep_sort.git /tmp/deep_sort
find /tmp/deep_sort/deep_sort -maxdepth 1 -name '*.py' ! -name '__init__.py' -exec cp {} src/ablation/DeepSORT/ \;
```

Afterwards the two directories should look like this — the adapters are tracked in git, everything else is downloaded:

```
src/ablation/
├── SORT/
│   ├── __init__.py
│   └── sort.py
└── DeepSORT/
    ├── __init__.py
    ├── deepsort.py          # adapter
    ├── detection.py
    ├── iou_matching.py
    ├── kalman_filter.py
    ├── linear_assignment.py
    ├── nn_matching.py
    ├── track.py
    └── tracker.py
```

### 4. Model Weights

The checkpoint and the exact configuration used for the paper are in `model/`:

| File | Description |
|---|---|
| `model/model.safetensors` | Weights in `safetensors` format |
| `model/config.yaml` | Frozen config (incl. dataset `mean`/`std`) matching the checkpoint |


---

## Project Structure

```
.
├── model/                        # Paper checkpoint + frozen config
│   ├── config.yaml
│   └── model.safetensors
└── src/
    ├── train.py                  # Entry point: training
    ├── evaluate.py               # Entry point: detection / tracking / event evaluation
    ├── analysis.py               # Entry point: lineage & fate analysis, paper figures
    ├── inference.py              # Detection + tracking loop
    ├── track_eval.py             # Standalone tracking-metric evaluation
    │
    ├── configs/
    │   └── config.yaml           # Default training configuration
    │
    ├── models/
    │   └── defdetr.py            # Deformable-DETR based detector
    │
    ├── dataset/                  # Dataloader and Dataset
    │
    ├── ablation/                 # Baseline trackers used in the ablation
    │
    ├── utils/
    │   ├── ...
    │   └── ctc/
    │       └── CTC_eval/         # <- place the CTC binaries here
    │
    ├── data/
    │   ├── dataset/              # <- place the dataset here
    │   ├── output/               # Default output folder of evaluate.py
    │   └── analysis/             # Default output folder of analysis.py
    │
    └── paper_eval/               # Result files backing the tables/figures in the paper
```

---

## Training

```bash
python src/train.py --name my_experiment
```

Hyperparameters, architecture, schedule, and output paths are all set in `src/configs/config.yaml`; there are no per-hyperparameter CLI flags. W&B logging is used — run `wandb login` first, or pass `--no-log` to skip W&B logging.

<details>
<summary><b>Arguments</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--name` | `str` | **required** | Run name. Used for the W&B run, the output directory and the saved config. |
| `--no-log` | flag | logging **on** | Pass to **disable** Weights & Biases logging. Takes no value. |

</details>

---

## Evaluation

Runs inference over every tracking video, then computes detection, tracking, and event metrics and writes one row per video to a CSV.

```bash
python src/evaluate.py \
    --checkpoint model/model.safetensors \
    --config model/config.yaml \
    --experiment_name overall
```

<details>
<summary><b>Arguments</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `str` | `model/model.safetensors` | Detector checkpoint to evaluate. |
| `--config` | `str` | `model/config.yaml` | Config matching the checkpoint (image type, channel, resize ratio). |
| `--coco` | `str` | `annotations.json` | Annotation filename inside `--data_dir`. |
| `--data_dir` | `str` | `src/data/dataset` | Dataset root. |
| `--output_dir` | `str` | `src/data/output` | Root for all evaluation output. |
| `--start_id` | `int` | `49` | First frame index (inclusive). See [Frame numbering](#frame-numbering). |
| `--end_id` | `int` | `234` | Last frame index (inclusive). |
| `--matching_alg` | `str` | `tt` | Association algorithm. `tt` (TransientTrack matching) additionally enables the CTC and division metrics. |
| `--ban_list` | `str` | – | CSV (single column, no header) of position IDs to exclude. |
| `--experiment_name` | `str` | `overall` | Output CSV is `<output_dir>/<experiment_name>.csv`. |

</details>

### Frame numbering

**Annotation space** — `--start_id` / `--end_id` refer to frame numbers as used in `annotations.json`. They select which annotated frames are evaluated and are inclusive on both ends. A video may be longer than its annotated range: in the dataset released with the paper, the videos run to frame 234 but annotation begins at frame 49, so evaluation uses `--start_id 49 --end_id 234`. For a dataset annotated from the first frame, use `--start_id 1 --end_id <N>`.

**CTC space** — the Cell Tracking Challenge ground truth carries its own origin, taken from the first `man_track*.tif` in `ctc_files/`. The CTC binaries pair predictions with ground truth by the number in the filename, so predicted masks and `res_track.txt` must use the same origin as the ground truth. This is detected automatically: `ctc_gt_origin()` reads it from the ground truth, and `combine_mask_labels` and `write_ctc_files` write predictions to match.

**If you are preparing your own ground truth**, the CTC specification numbers each sequence from `000`. Frame numbers inside `man_track.txt` (columns 2 and 3) must use the same origin as the `man_track*.tif` filenames — a mismatch between the two is not detectable automatically and will produce wrong scores rather than an error. The number of ground-truth frames must also equal the number of evaluated frames; `shift_files` asserts this.

### Output layout

```
src/data/output/
├── overall.csv                          # one row per video, all metrics
└── evaluation/
    └── <position>/
        ├── summary.json                 # tracklets, lineage, fates -> input to analysis.py
        ├── ctc_files/                   # predicted tracks in CTC format
        └── eval_ctc.csv                 # raw CTC measure output
```

### Metrics in `overall.csv`

| Group | Columns | Source |
|---|---|---|
| Tracking | `HOTA`, `DetA`, `AssA`, `MOTA`, `MOTP`, `IDF1` | TrackEval |
| Cell Tracking Challenge | `DET`, `TRA`, `LNK` | CTC binaries (`--matching_alg tt` only) |
| Detection | `mAP`, `AP50`, `AP_alive`, `AP_dead` | `torchmetrics` `MeanAveragePrecision` |
| Apoptosis | `Precision`, `Recall`, `F1`, `TP`, `FP`, `FN`, `TN`, `Mean-Error`, `Timing Errors`, `Timing Excluded` | `apoptosis_evaluation` |
| Division | `DIV_nGT`, `DIV_nPred`, `DIV_WC`, `DIV_TP_i0`, `DIV_TP_i1`, `DIV_TP_i2` | `traccuracy` via `BoxIOUMatcher` |
| Runtime | `Inference-Time` | seconds per video |

<details>
<summary><b>How the event metrics are defined</b></summary>

**Division.** Only the minimal sufficient statistics are logged, not derived rates. `DIV_TP_i{n}` is the number of correctly detected divisions allowing a frame buffer of `n`; `DIV_WC` counts divisions where the lineage link is wrong and does *not* relax with the buffer. Rates are obtained by pooling raw counts across videos (micro-averaging) rather than averaging per-video F1, which avoids undefined values for videos with zero predicted divisions:

```
Recall(i)    = sum(DIV_TP_i) / sum(DIV_nGT)
Precision(i) = sum(DIV_TP_i) / sum(DIV_nPred)
Assoc. acc.  = sum(DIV_TP_i0) / (sum(DIV_TP_i0) + sum(DIV_WC))
```

`i` is the frame buffer: a predicted division counts as detected if it falls within `i` frames of the ground-truth division. Precision and recall are reported as BC(i).

**Apoptosis.** `delta` (default `5`) is the tolerance of the timing check; `window` (default `2`) is the temporal matching window. They are independent parameters, so the reported mean localisation error can exceed `window`. Unmatched tracks are excluded from the denominators.

</details>

---

## Analysis

Consumes the `summary.json` files produced by evaluation and generates the lineage-level statistics and paper figures.

```bash
python src/analysis.py
```

The first run extracts per-cell mean intensity and writes it back into the `summary.json` files; this is the slow part and only needs to happen once. Pass `--skip_intensity` on every subsequent run.

<details>
<summary><b>Arguments</b></summary>

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data_dir` | `str` | `src/data/output/evaluation` | Directory of per-position evaluation output. |
| `--output_dir` | `str` | `src/data/analysis/results` | Where figures are written. |
| `--num_frames` | `int` | `234` | Total frames per video. |
| `--skip_intensity` | flag | off | Skip the intensity-extraction pass. |

</details>

---

## Using TransientTrack on Your Own Data

Nothing in the method assumes fluctuating signals — the detector needs annotated boxes, and the tracker needs the detector's embeddings. Adapting it to another dataset means retraining the detector and adjusting a handful of config values.

What you need to supply:

- **Annotations** in the CocoVid format read by `dataset/parser.py`: a `annotations.json` with `videos`, `images` (each carrying `video_id` and a frame index), and `annotations` with `bbox`, `category_id` (living / dead) and `cell_id`. One directory of frames per video,   named by position ID.
- **CTC ground truth** per position in `ctc_files/`, only if you want DET / TRA / LNK scores. Any frame origin works — it is read from the ground truth and the predictions are written to match (see [Frame numbering](#frame-numbering)).

What to change in `src/configs/config.yaml`:

| Field | Meaning |
|---|---|
| `misc.img_type` | `rgb` or `gray`; `misc.channel` selects a single channel |
| `training.resize_ratio` | Input scaling; the CTC masks are written back at this scale |
| `training.mean` / `std` | Used by the dataloader and written into the saved config |
| `misc.iou_threshold`, `low_conf_upper`, `low_conf_lower` | Detection NMS and the two-stage confidence split |

Handled automatically, with no configuration needed: `--start_id` / `--end_id` default to the annotated range, and the CTC frame origin is taken from the ground truth.

The released checkpoint is trained on U-2 OS cells at 1024x1024 with three fluorescent channels. Expect to retrain for a different cell line, magnification, or channel count; the association stage itself needs no retraining, since it has no learned parameters.

---

## Reproducing the Paper Results

```bash
# 1. Evaluate the released checkpoint
python src/evaluate.py \
    --checkpoint model/model.safetensors \
    --config model/config.yaml \
    --experiment_name overall

# 2. Analysis + figures
python src/analysis.py
```

The reference numbers for every table and figure are committed under `src/paper_eval/`.

---

## Citation

If you use the data, please cite both the dataset and the paper:

```bibtex
@misc{buerger2025transienttrack,
      title={TransientTrack: Advanced Multi-Object Tracking and Classification of Cancer Cells with Transient Fluorescent Signals}, 
      author={Florian Bürger and Martim Dias Gomes and Nica Gutu and Adrián E. Granada and Noémie Moreau and Katarzyna Bozek},
      year={2025},
      eprint={2512.01885},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01885}, 
}
```

```bibtex
@misc{buerger2025dataset,
  author    = {Bürger, Florian and Dias Gomes, Martim and Gutu, Nica and Granada, Adrián E. and Moreau, Noémie and Bozek, Katarzyna},
  title     = {Large-Scale Time-Lapse Microscopy Dataset for Cell Tracking},
  publisher = {PUBLISSO},
  year      = {2025},
  doi       = {10.4126/FRL01-006526608},
  url       = {https://repository.publisso.de/resource/frl:6526608},
  note      = {Dataset, CC BY 4.0}
}
```
---

## License

This project is released under the [MIT License](LICENSE). The dataset is published separately under CC BY 4.0.


### Third-party code
 
| Component | Source | License |
|---|---|---|
| `utils/ctc/ctc_eval.py` | [EmbedTrack](https://github.com/kaloeffler/EmbedTrack) | MIT |
| `dataset/parser.py` | [mmtracking](https://github.com/open-mmlab/mmtracking) | Apache-2.0 |
| `ablation/ByteTrack/` | [ByteTrack](https://github.com/FoundationVision/ByteTrack) | MIT (modified: `np.float` replaced for NumPy >= 1.24) |
| `ablation/DeepSORT/deepsort.py` | ByteTrack / [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch) | MIT |
 
Not redistributed, downloaded by the user:
 
| Component | Source | License |
|---|---|---|
| `ablation/SORT/sort.py` | [SORT](https://github.com/abewley/sort) | GPL-3.0 |
| `ablation/DeepSORT/` (upstream) | [deep_sort](https://github.com/nwojke/deep_sort) | GPL-3.0 |
| `utils/ctc/CTC_eval/` | [Cell Tracking Challenge](http://celltrackingchallenge.net/evaluation-methodology/) | CTC terms |
