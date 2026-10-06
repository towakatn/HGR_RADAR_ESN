# Hand Gesture Recognition from Doppler Radar Signals Using Echo State Networks

This project classifies hand gestures captured by radar sensors using reservoir computing with an Echo State Network (ESN).  
It evaluates multiple readout methods on two radar datasets: **Dop-NET** and **Soli**.

---


## Project Structure

```
crest/
├── README.md
├── requirements.txt
├── modules/                  # Shared implementation
│   ├── data_loaders.py        # Dop-NET and Soli data loaders
│   ├── reservoir_computer.py  # Dop-NET sparse ESN
│   ├── reservoir.py           # Soli dense ESN
│   ├── readouts.py            # Multi/single ESN readout models
│   ├── classifiers.py         # RF, SVM, and Ridge factories
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
│   ├── soli_config.py         # Dataset/model/classifier settings
│   ├── separate_channel_dtm_converter.py
│   ├── separate_channel_rtm_converter.py
│   ├── SoliData/dsp/          # Raw data (.h5)
│   ├── DTM/                  # Doppler-Time Map per channel
│   └── RTM/                  # Range-Time Map per channel
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

- Python 3.8+

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
single_reservoir_config=None)` accept replacement configuration dictionaries.
Soli's `get_methods(...)` builds the seven readout classes and their arguments;
`run_soli_evaluation(...)` can also evaluate an individual model.

Imports previously under `Dop-NET/modules` or `Soli/modules` now come from
`crest/modules`. Use `modules.data_loaders`, `modules.readouts`, and
`modules.evaluation` from `crest/`.

Run regression tests from `crest/` with the project environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

## Citation
T. Sano and G. Tanaka, Hand Gesture Recognition from Doppler Radar Signals Using Echo State Networks, International Joint Conference on Neural Networks (WCCI 2026), accepted

Preprint available on arXiv: https://arxiv.org/abs/2602.04436v1