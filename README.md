# Training American Sign Language Alphabets

Trains an MLP to classify the ASL alphabet from MediaPipe hand keypoints.
Keypoints are extracted once and cached, then reused by every training run.

## Setup

**1. Make sure your SSH key is connected to GitHub.**

```bash
ssh -T git@github.com
```

You should see `Hi <username>! You've successfully authenticated`. If not, follow
[GitHub's SSH key guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

**2. Clone the repository.**

```bash
git clone git@github.com:PaulMsk141/SIgn-Language-.git SignLanguageDetection
```

**3. Enter the project.**

```bash
cd SignLanguageDetection
```

**4. Get the dataset.** The code expects it at `datasets/`, so clone it under that name.

```bash
oxen clone https://hub.oxen.ai/nex-team-inc/SignLanguageDataset datasets
```

<details>
<summary>Fallback, if that repo is not uploaded yet</summary>

```bash
oxen clone https://hub.oxen.ai/PaulMsk/signlanguagedataset datasets
```

This mirror ships sources 15 and 16 as raw video, so turn them into frames
(`datasets/video_frames/<CLASS>/<source>_<clip>_f<frame>.jpg`, the layout the
notebooks read):

```bash
python extract_video_frames.py
```

It also has no captured evaluation set, so clone the test capture at the repository root:

```bash
oxen clone https://hub.oxen.ai/PaulMsk/signlanguagetestdata signlanguagetestdata
```

</details>

The current pipeline also expects these repository-root folders:

```text
validation_set/          # left/right MP4 captures; independent model selection
fine-tune/               # left/right/{A..Z,none}/, 100 numbered stills per folder
signlanguagetestdata/    # left/right test capture; kept untouched until final evaluation
```

**5. Install the Python dependencies.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy opencv-python matplotlib pandas scikit-learn tqdm ipykernel optuna
```

That is what the core notebooks and optional tuning runner import. `requirements.txt` covers the
whole repo, including the older TensorFlow image models, and is not needed here.

`_hand_models/` is already in the repo. It holds the pretrained MediaPipe handpose
weights (palm detection + landmark extraction) as TorchScript, loaded with
`torch.jit.load`, so there is nothing to download and no `mediapipe` install.

## Core workflow

Run the core notebooks in this order.

| # | Notebook | What it does |
|---|---|---|
| 1 | `extract_keypoints.ipynb` | Filters the public images into `train_*`, materializes the independent validation/test captures, and writes raw landmarks and detector scores to `keypoints.npz`. |
| 2 | `experiment_mlp.ipynb` | Builds per-sample-normalized features, trains the MLP, validates on `validation_set/`, evaluates the untouched test capture, and writes `weights/<FEATURES>_mlp.pt` plus `runs/<FEATURES>.npz`. |
| 3 | `experiments.ipynb` | Reads completed run files and plots coverage, learning curves, per-class results, and confusion matrices. Nothing is trained here. |

Run all cells top to bottom. Extraction also writes `palm_rois.npz`,
`landmarks.npz`, and sampled `validation_set_frames/`. Their content-aware keys
reuse unchanged data, so adding or replacing one capture does not re-extract the
public training images.

All adjustable MLP settings are in the first configuration cell of
`experiment_mlp.ipynb`. Supported feature modes are `2d`, `3d`, and `2d+3d`;
image-landmark depth/“2.5d” is deliberately excluded. Every image normalizes its
selected 2D and world-3D blocks independently to `[0,1]`.

## Optional hyperparameter tuning

The tuning notebook is a results viewer; run the search separately:

```bash
python run_new_pipeline_cv_tune.py --time-budget 3600 --jobs 4
```

Open `new_pipeline_cv_hyper_param_tune.ipynb` to inspect trial rankings,
group-aware cross-validation accuracy, the independent captured-validation score,
generalization gaps, and parameter importance. The test capture is never loaded by
the tuner.

## Fine-tuning or adding `none`

After producing `weights/2d+3d_mlp.pt`, run `fine-tuned-mlp.ipynb`. Its first
configuration cell contains every data, gate, optimization, cache, and preview
setting.

```python
INCLUDE_NONE = False  # ordinary 26-class A-Z fine-tuning
INCLUDE_NONE = True   # transfer learning: expand the head to A-Z + none
```

The default uses all 100 images from every selected `fine-tune/<hand>/<class>/`
folder. Transfer mode copies the pretrained hidden/A-Z weights, initializes a new
trainable `none` output, and uses separate `ALL_NONE_1-100` caches and checkpoints.
Validation and test remain A-Z; a `none` prediction on either counts as an error.
