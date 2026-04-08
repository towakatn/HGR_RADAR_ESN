# Hand Gesture Recognition from Doppler Radar Signals Using Echo State Networks

This project classifies hand gestures captured by radar sensors using reservoir computing with an Echo State Network (ESN).  
It evaluates multiple readout methods on two radar datasets: **Dop-NET** and **Soli**.

---


## Project Structure

```
HGR_Radar/
├── README.md
├── requirements.txt
├── Dop-NET/                  # For the Dop-NET dataset
│   ├── run_all.py            # Run all readout evaluations
│   ├── Data/
│   │   ├── Training Data/    # Training data for subjects A-F (.mat)
│   │   └── Test Data/        # Test data (.mat)
│   └── modules/
│       ├── config.py          # Parameter settings
│       ├── data_loader.py     # MATLAB data loader
│       ├── reservoir_computer.py  # ESN reservoir
│       ├── evaluation.py      # Evaluation pipeline
│       ├── RF.py              # Random Forest readout
│       ├── SVM.py             # SVM readout
│       └── Ridge.py           # Ridge readout
│
└── Soli/                     # For the Google Soli dataset
    ├── run_all.py            # Run all readout evaluations
    ├── separate_channel_dtm_converter.py  # DTM conversion
    ├── separate_channel_rtm_converter.py  # RTM conversion
    ├── SoliData/dsp/         # Raw data (HDF5)
    ├── DTM/                  # Doppler-Time Map (per channel)
    ├── RTM/                  # Range-Time Map (per channel)
    └── modules/
        ├── config.py              # Parameter settings
        ├── dataloader.py          # Dual data-type loader
        ├── reservoir.py           # Variable-length ESN reservoir
        ├── evaluation.py          # Evaluation pipeline
        ├── multi_*.py             # Readouts for multi-reservoir setup
        ├── single_*.py            # Readouts for single-reservoir setup
        └── multi_feat_esn_readout.py
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

## Directory Details

### `Dop-NET/modules/`

| File | Role |
|---------|------|
| `config.py` | Defines parameters for reservoir, classifiers, and evaluation |
| `data_loader.py` | Loads MATLAB `.mat` files and converts them to spectrograms |
| `reservoir_computer.py` | ESN implementation (sparse weight generation and state update) |
| `evaluation.py` | Common pipeline for four evaluation protocols |
| `RF.py` / `SVM.py` / `Ridge.py` | Factory functions for each readout classifier |

### `Soli/modules/`

| File | Role |
|---------|------|
| `config.py` | Defines parameters for both multi and single settings |
| `dataloader.py` | Loads 4-channel DTM + RTM data |
| `reservoir.py` | Variable-length time-series ESN (`VariableLengthESN`) |
| `single_reservoir.py` | Wrapper for single-reservoir mode |
| `evaluation.py` | Evaluation pipeline |
| `multi_RR_L.py` / `multi_RR_N.py` | Ridge-regression readouts (linear / nonlinear) |
| `multi_SVM.py` / `multi_RF.py` | SVM / RF readouts for multi-reservoir mode |
| `single_RF.py` / `single_SVM.py` / `single_Ridge.py` | Readouts for single-reservoir mode |
| `multi_feat_esn_readout.py` | Multi-feature ESN readout |
| `multi_classifier_readout.py` | Multi-classifier readout |
