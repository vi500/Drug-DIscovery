# Module 3 Full Package

This package contains the complete requested source layout and pipeline scripts.

## Installation
Recommended on Windows:
```powershell
cd "D:\Major project\drug_safety_model"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
Copy the package contents into `D:\Major project\drug_safety_model`, then:
```powershell
python code\08_run_pipeline.py --ligands sample_ligands.csv --affinity sample_affinity.csv --name test_run --seed 42
```

For a real project run, provide your real ligand and affinity CSVs.

## Important
No fake model weights, predictions, labels, or benchmark numbers are included. ADMET-AI downloads/loads its actual pretrained weights when installed, and PyTDC downloads the real datasets. If those components cannot load, the relevant stage fails rather than silently substituting fake predictions.

The custom DNN checkpoint is created only after Step 04 actually trains successfully.


## Python 3.13 note
This package is configured for Python 3.13. The original PyTDC release declares a legacy numpy<2 constraint, which conflicts with Python 3.13 wheels. Install the main requirements first; if PyTDC is needed, install `PyTDC==1.1.15` with `--no-deps` and use its runtime dependencies from requirements.txt. The pipeline never fabricates endpoint values and will stop if required ADMET-AI endpoints are absent.
