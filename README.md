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

It also has no evaluation set, so clone that one inside `datasets/`:

```bash
cd datasets && oxen clone https://hub.oxen.ai/PaulMsk/signlanguagetestdata && cd ..
```

</details>

**5. Install the Python dependencies.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch matplotlib scikit-learn tqdm
```

`_hand_models/` is already in the repo. It holds the pretrained MediaPipe handpose
weights (palm detection + landmark extraction) used during keypoint extraction, so
there is nothing to download.

## Training

Run the notebooks in this order.

| # | Notebook | What it does |
|---|---|---|
| 1 | `extract_keypoints.ipynb` | Runs the handpose model over every image and caches the keypoints to `keypoints.npz`. Slow, but only once. |
| 2 | `experiment_mlp.ipynb` | Trains the MLP on that cache and evaluates it. Best weights go to `mlp_best.pt`. |
| 3 | `experiments.ipynb` | Detailed evaluation: per-epoch curves, per-class precision/recall/F1, confusion matrix, and a side-by-side comparison of feature modes. |

Run all cells in each, top to bottom. Step 1 also writes the resumable caches
`palm_rois.npz` and `landmarks.npz`, so an interrupted run picks up where it stopped.

To train on a different feature mode (`2d`, `2.5d`, `3d`, `2d+3d`, `2.5d+3d`), change
`FEATURES` in `experiment_mlp.ipynb` and rerun it. The cache does not need rebuilding:
`keypoints.npz` stores raw landmarks, and the feature mode is applied on load.
