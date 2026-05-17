# CryptoBackdoor
This repository contains the official code for the paper "Rethinking the Stealthiness of Cryptographically Undetectable Backdoors in Practical RFF Learning", accepted by KDD 2026.


## Environment

```bash
conda create --name cryptobackdoor python=3.9
conda activate cryptobackdoor
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scikit-learn scipy matplotlib tqdm cvxpy
```

CIFAR-10 is downloaded automatically. For Covertype, place the data file at:

```text
./dataset/covtype.data
```

## Directory

```text
CryptoBackdoor1/
├── README.md
├── detect.py
└── src/
    ├── cifar10_acc.py
    ├── cifar10_clwe.py
    ├── covertype_acc.py
    └── covertype_clwe.py
```

## Files

### `src/*_acc.py`

Compares accuracy under different preprocessing choices such as raw input, standardization, and scaling on CIFAR-10 and Covertype.


### `src/*_clwe.py`

Runs the CLWE/Goldwasser-style backdoor attack on CIFAR-10 and Covertype. It trains an RFF classifier, constructs a malicious feature matrix and trigger vector `bk`, then reports clean accuracy, triggered accuracy, and prediction flip ratio.


### `detect.py`

Evaluates a detector for triggered samples. It marks a sample as suspicious when its infinity norm exceeds a threshold and reports detection rate and false positive rate under different preprocessing settings.

## Workflow

```bash
mkdir -p dataset figs

python src/cifar10_acc.py
python src/covertype_acc.py

python src/cifar10_clwe.py
python src/covertype_clwe.py

python detect.py
```