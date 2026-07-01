# DDM2 for Mayo Low-Dose CT Denoising

Self-supervised denoising of Mayo low-dose abdominal CT, adapted from **DDM2** (*Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models*, ICLR 2023). The original DDM2 3-stage pipeline is applied to CT: self-supervision comes from **odd/even projection (sinogram-split) reconstructions** of the same scan, which form the noisy–noisy training pairs. Training and evaluation are restricted to the abdominal slice band **150–200** on the high-noise Mayo dataset.

## How it works (3-stage DDM2 pipeline)

DDM2 turns a paired denoiser into a diffusion model in three stages:

1. **Stage 1 – Noise model (`train_noise_model.py`).** Trains the initial noise-representation network from the noisy data.
2. **Stage 2 – State matching (`match_state.py`).** Matches each noisy sample to a diffusion timestep, producing a `stage2_matched.txt` state file used to condition stage 3.
3. **Stage 3 – Diffusion model (`train_diff_model.py`).** Trains the conditional diffusion denoiser (`which_model_G: "mri"`, U-Net inner_channel 32, `[1,2,4,8,8]`) using the stage-2 states.

Self-supervision is faithful to DDM2: the noisy–noisy pair is the **odd** vs **even** independent reconstruction of the same CT scan (`simulation_file_odd` / `simulation_file_even`), with `simulation_file_all` as the full-dose-equivalent input. An optional Noise2Noise (N2N) teacher can be referenced for comparison.

## Repository structure

- `train_noise_model.py` — Stage 1: train the DDM2 noise model.
- `match_state.py` — Stage 2: state-matching, emits the `stage2_matched.txt` state file.
- `train_diff_model.py` — Stage 3: train the conditional diffusion denoiser.
- `inference_mayo.py` — Run inference for one patient; writes full-volume `ddm2_first_step.nii.gz` and `ddm2_final.nii.gz`.
- `eval_mayo.py` — Compute MAE / SSIM / LPIPS vs GT (and N2N baseline); writes results Excel.
- `denoise.py`, `sample.py`, `test.py` — Original DDM2 denoise/sample/test utilities.
- `config/mayo_ct_denoise.json` — Main config (default patient `L192`, slice_range `[150,200]`, HU window `[-200,250]`).
- `config/mayo_ct_denoise_L291.json`, `config/mayo_ct_denoise_L310.json` — Per-patient configs (differ only in `patient_ids` and the `stage2_file` path).
- `data/mayo_ct_dataset.py` — `MayoCTDataset`: reads the Excel manifest, filters by batch/`patient_ids`, builds odd/even pairs, applies slice_range and HU normalization.
- `data/` — dataset package (`__init__.py`, `prepare_data.py`, `util.py`).
- `model/` — DDM2 model code (`model.py`, `model_stage1.py`, `noise_model.py`, `networks.py`, `base_model.py`, `mri_modules/`).
- `core/` — `logger.py`, `metrics.py`.
- `nii2npy.py` — Optional: convert `.nii.gz` volumes to `.npy` to speed up training/inference.
- `metrics.py`, `quantitative_metrics.ipynb` — Metric helpers / analysis notebook.
- `run_stage1.sh`, `run_stage2.sh`, `run_stage3.sh` — Stage launchers.
- `environment.yml` — Conda environment (`ddm2`).
- `experiments/` — Per-patient run outputs (`L192/`, `L291/`, `L310/`), each with `checkpoint/` and `inference/`.
- `mayo_results_L291.xlsx`, `mayo_results_L310.xlsx`, `mayo_ddm2_results.xlsx` — Saved metric results.

## Setup

```bash
conda env create -f environment.yml
conda activate ddm2
```

## How to run

Update paths in the chosen `config/*.json` (and in the `EXCEL_PATH` / `GT_ROOT` / `N2N_ROOT` constants at the top of `eval_mayo.py`) to your local layout before running. Paths use the `/host/d/...` docker convention (`/host/d` = `D:\`); for native Windows set `DATA_ROOT` accordingly.

**Stage 1 — noise model**
```bash
python3 train_noise_model.py -p train -c config/mayo_ct_denoise.json
# or: ./run_stage1.sh   (note: run_stage1.sh points at the older config/ct_denoise.json)
```

**Stage 2 — state matching** (produces `stage2_file`)
```bash
mkdir -p experiments/mayo_ct_denoise_teacher
python3 match_state.py -p train -c config/mayo_ct_denoise.json
# or: ./run_stage2.sh
```

**Stage 3 — diffusion training**
```bash
python3 train_diff_model.py -p train -c config/mayo_ct_denoise.json
# or: ./run_stage3.sh
```

**Inference** (writes `ddm2_first_step.nii.gz` and `ddm2_final.nii.gz` per patient under `experiments/*mayo*/inference/<patient_id>/`; auto-selects the latest matching checkpoint)
```bash
python inference_mayo.py -c config/mayo_ct_denoise.json --patient_id L192
# or by index within the selected batch:
python inference_mayo.py -c config/mayo_ct_denoise.json --patient_idx 0
```

**Evaluation** (MAE / SSIM / LPIPS vs GT, with N2N baseline; auto-finds the newest `inference/` dir)
```bash
python eval_mayo.py --patient_id L192
# all val + test patients:
python eval_mayo.py --batch val test
```

### Per-patient configs
`L291` and `L310` have dedicated configs identical to the main one except for `patient_ids` and the stage-2 state path (`stage2_L291.txt` / `stage2_L310.txt`). Point stages 2/3 and inference at the corresponding config, e.g.:
```bash
python3 match_state.py -p train -c config/mayo_ct_denoise_L291.json
python3 train_diff_model.py -p train -c config/mayo_ct_denoise_L291.json
python inference_mayo.py -c config/mayo_ct_denoise_L291.json --patient_id L291
```

## Data layout & key specifics

- **Manifest (Excel):** `mayo_low_dose_CT_highnoise_v2_ddm2.xlsx` with columns `Patient_ID`, `batch`, `simulation_file_odd`, `simulation_file_even`, `simulation_file_all`, `ground_truth_file`. The dataset filters rows by `batch` (`train_batches` / `val_batches`, here `["test"]`) and by `patient_ids`.
- **Self-supervision:** noisy–noisy pair = odd vs even sinogram-split reconstructions of the same scan (faithful DDM2 setup).
- **Slice range:** training/eval restricted to slices **150–200** (`datasets.*.slice_range = [150, 200]`; `eval_mayo.py` `EVAL_SLICE_START/END = 150/200`). GT full volumes are cropped to this band at evaluation.
- **HU / normalization:** training/inference normalize with HU window **[-200, 250]** (`HU_MIN`/`HU_MAX`); images are scaled to `[-1, 1]` and mapped back to HU on output.
- **Evaluation window:** metrics computed only within the reference-image HU window **[-160, 240]** (abdominal CT), masking out-of-window pixels.
- **GT:** per-patient full volume `img.nii.gz` under `GT_ROOT` (`.../mayo/original_imgs/<patient_id>/`).
- **N2N baseline (optional):** `.../noise2noise数据/<patient_id>/random_0/epoch50/pred_img.nii.gz` (missing patients are skipped).
- **Image size:** 512×512; single channel; diffusion `n_timestep = 1000`, `rev_warmup70` schedule.
