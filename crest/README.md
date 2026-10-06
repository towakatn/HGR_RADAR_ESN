# Hand Gesture Recognition from Doppler Radar Signals Using Echo State Networks

This project classifies hand gestures captured by radar sensors using reservoir computing with an Echo State Network (ESN).  
It evaluates multiple readout methods on **Dop-NET** and **Soli**, and compares
reservoir topology and fusion stage on the 16-bit **HAR-mmWave** dataset.

---


## Project Structure

```
crest/
├── README.md
├── HAR_EXPERIMENTS.md         # Same-room, same-distance 16-bit HAR comparison
├── requirements.txt
├── modules/                  # Shared implementation
│   ├── data_loaders.py        # Dop-NET and Soli data loaders
│   ├── reservoir_computer.py  # Dop-NET sparse ESN
│   ├── reservoir.py           # Soli dense ESN
│   ├── readouts.py            # Multi/single ESN readout models
│   ├── classifiers.py         # RF, SVM, and Ridge factories
│   ├── fusion.py              # Controlled reservoir/fusion architectures
│   ├── fusion_evaluation.py   # Paired splits, shared search, and result exports
│   ├── har_data.py            # 16-bit HAR room selection and DTM/RTM projection
│   ├── har_download.py        # Fetch one room with LFS size/SHA-256 verification
│   ├── har_config.py          # Shared same-room, same-distance HAR settings
│   ├── har_experiment.py      # HAR comparison runner and split audit exports
│   ├── evaluation.py          # Dataset-specific evaluation protocols
│   └── converters.py          # Soli DTM/RTM conversion
├── Dop-NET/
│   ├── run_all.py             # Pass Dop-NET settings to shared modules
│   ├── dopnet_config.py       # Dataset/model/classifier settings
│   └── Data/
│       ├── Training Data/     # Subjects A-F (.mat)
│       └── Test Data/
├── Soli/
│   ├── run_all.py             # Pass Soli settings to shared modules
│   ├── run_fusion.py          # Controlled topology/fusion comparison
│   ├── soli_config.py         # Dataset/model/classifier settings
│   ├── separate_channel_dtm_converter.py
│   ├── separate_channel_rtm_converter.py
│   ├── SoliData/dsp/          # Raw data (.h5)
│   ├── DTM/                  # Doppler-Time Map per channel
│   └── RTM/                  # Range-Time Map per channel
├── HAR-Dataset-Project/       # Local dataset checkout; data and outputs are ignored
│   ├── Human activity recognition V2.0_Clipping/  # 16-bit tensors
│   └── results/              # Within-room comparisons and train/test manifests
└── tests/                    # Regression tests
```


## Dataset Downloads

You can download the datasets used in this repository from the following public pages:

- Dop-NET (official repository): https://github.com/UCLRadarGroup/DopNet
- Dop-NET (dataset distribution page): https://rdr.ucl.ac.uk/articles/dataset/Dop-Net_Data/25486597
- Soli (public dataset example / Kaggle): https://www.kaggle.com/datasets/chandragupta0001/soli-data

---



## Setup

### Prerequisites

- Python 3.12+

### Installation

```bash
git clone <repository-url>
cd HGR_Radar
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### Dependencies

```
h5py
matplotlib
numpy
tqdm
scikit-learn
seaborn
```

---

## Usage

### Dop-NET

```bash
cd Dop-NET
python run_all.py
```


### Soli

#### 1. Data Preprocessing (First Time Only)

Convert raw data in `SoliData/dsp/` into DTM / RTM.

```bash
cd Soli
python separate_channel_dtm_converter.py   # DTM conversion
python separate_channel_rtm_converter.py   # RTM conversion
```

#### 2. Run Evaluation

```bash
cd Soli
python run_all.py
```
---

## Shared Modules and Settings

Edit `Dop-NET/dopnet_config.py` or `Soli/soli_config.py` to change dataset paths,
reservoir settings, or classifier settings. Each `run_all.py` passes those values
to the shared modules. Default paths are relative to the dataset directory, so
scripts can also be run from `crest/`:

```bash
python Dop-NET/run_all.py
python Soli/run_all.py
python Soli/separate_channel_dtm_converter.py
python Soli/separate_channel_rtm_converter.py
```

The two ESN implementations retain their original weight initialization and
random-number behavior. Dop-NET keeps a one-direction 50:50 split with seed
`reservoir_seed + 1` and a new reservoir per CV fold. Soli keeps its two-direction
50:50 split with seed 42 and its original model initialization order. Both keep
their original session and subject splits, result keys, and method order.

For Python callers, Dop-NET's `main(data_config=None, reservoir_config=None)`
and Soli's `main(data_config=None, multi_reservoir_config=None,
single_reservoir_config=None, include_fusion=True, fusion_config=None)` accept
replacement configuration dictionaries. Use `include_fusion=False` to run
only the original seven methods from Python.
Soli's `get_methods(...)` builds the seven readout classes and their arguments;
`run_soli_evaluation(...)` can also evaluate an individual model.

Imports previously under `Dop-NET/modules` or `Soli/modules` now come from
`crest/modules`. Use `modules.data_loaders`, `modules.readouts`, and
`modules.evaluation` from `crest/`.

Run regression tests from `crest/` with the project environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

## Controlled Soli Fusion Experiments

Soli also compares single-map baselines, Single–Early, Parallel–Early,
Parallel–Intermediate, and Parallel–Late with a shared total-node budget,
the existing linear Ridge regression (RR_L), regularization, data splits, and
hyperparameter candidates. From `crest/`:

```bash
python Soli/run_fusion.py
python Soli/run_fusion.py --quick
```

`Soli/run_all.py` runs this comparison after the original seven methods;
`python Soli/run_all.py --legacy-only` runs only those original methods.
See [Soli/FUSION_EXPERIMENTS.md](Soli/FUSION_EXPERIMENTS.md) for the protocols,
Late fusion rules, fairness constraints, output files, and research references.

## Same-Room and Same-Distance HAR Fusion Experiments

The HAR runner selects only high-precision recordings with one filename room
identifier and one distance identifier. The defaults H1 and D1 (1.5 m) select
600 recordings, with 300 training and 300 test samples per direction.
It applies the same five comparison families as Soli, with 400 total
nodes, existing RR_L readouts, regularization 0.1, zero bias, and seeds 42/43/44.
The default protocol is a stratified 50:50 sample split evaluated in both
directions. Subjects can occur in both partitions; this measures recognition
within the selected room and distance rather than generalization to new subjects,
rooms, or distances.

From `crest/`, fetch the selected room if the checkout contains Git LFS pointers,
then run the comparison:

```bash
.venv/bin/python -m modules.har_download --room 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m modules.har_experiment --room 1 --distance 1
```

The implementation and configuration live in tracked shared modules, so the
experiment does not depend on scripts inside the ignored local dataset checkout.
Outputs go to `HAR-Dataset-Project/results/` and include the input manifest,
per-recording train/test membership, CSV summaries, and a plot. See
[HAR_EXPERIMENTS.md](HAR_EXPERIMENTS.md) for preprocessing, settings, and usage.

## Citation
T. Sano and G. Tanaka, Hand Gesture Recognition from Doppler Radar Signals Using Echo State Networks, International Joint Conference on Neural Networks (WCCI 2026), accepted

Preprint available on arXiv: https://arxiv.org/abs/2602.04436v1
