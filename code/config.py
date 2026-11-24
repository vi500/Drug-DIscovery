from pathlib import Path
import os
import random
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Major project\safety_module")

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STRUCTURES_DIR = PROJECT_ROOT / "structures"
DOCS_DIR = PROJECT_ROOT / "docs"
RUNS_DIR = PROJECT_ROOT / "runs"
CODE_DIR = PROJECT_ROOT / "code"
TESTS_DIR = PROJECT_ROOT / "tests"


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# ENDPOINTS
# ============================================================

ENDPOINTS = (
    "AMES",
    "hERG",
    "DILI",
    "ClinTox",
    "Carcinogens",
    "BBB",
    "HIA",
    "Solubility",
)

CLASSIFICATION_ENDPOINTS = (
    "AMES",
    "hERG",
    "DILI",
    "ClinTox",
    "Carcinogens",
    "BBB",
    "HIA",
)

REGRESSION_ENDPOINTS = (
    "Solubility",
)


# ============================================================
# MOLECULAR FEATURES
# ============================================================

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048

USE_RDKIT_DESCRIPTORS = True

DNN_DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "MolMR",
    "FractionCSP3",
)


# ============================================================
# DNN PARAMETERS
# ============================================================

HIDDEN = (512, 256)

DROPOUT = 0.30

LR = 1e-3

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64

EPOCHS = 100

PATIENCE = 12


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10


# ============================================================
# HARD FILTER THRESHOLDS
# ============================================================

AMES_HARD_FAIL_THRESHOLD = 0.50

HERG_HARD_FAIL_THRESHOLD = 0.50


# ============================================================
# STRUCTURAL ALERTS
# ============================================================

PAINS_ENABLED = True

BRENK_ENABLED = True

CNS_TARGET_DEFAULT = False


# ============================================================
# SAFETY SCORE WEIGHTS
# ============================================================

W_QED = 0.20

W_LIPINSKI = 0.10

W_ADME = 0.25

W_TOX = 0.35

W_ALERT = 0.10


# ============================================================
# REWARD WEIGHTS
# ============================================================

W_AFFINITY = 0.50

W_SAFETY = 0.50


# ============================================================
# RECOMMENDATION
# ============================================================

SHORTLIST_SIZE = 50

MIN_RECOMMEND_SAFETY = 0.50


# ============================================================
# STEP 1 OUTPUTS
# ============================================================

VALIDATED_FILE = (
    OUTPUTS_DIR / "validated_candidates.csv"
)

INVALID_FILE = (
    OUTPUTS_DIR / "invalid_smiles.csv"
)

DUPLICATES_FILE = (
    OUTPUTS_DIR / "duplicate_smiles.csv"
)


# ============================================================
# STEP 2 OUTPUT
# ============================================================

DRUG_LIKENESS_FILE = (
    OUTPUTS_DIR / "drug_likeness.csv"
)


# ============================================================
# STEP 3 OUTPUT
# ============================================================

ADMET_AI_FILE = (
    OUTPUTS_DIR / "admet_ai_predictions.csv"
)


# ============================================================
# STEP 4 OUTPUTS
# ============================================================

DNN_PREDICTIONS_FILE = (
    OUTPUTS_DIR / "dnn_candidate_predictions.csv"
)

DNN_METRICS_FILE = (
    OUTPUTS_DIR / "dnn_metrics.csv"
)


# ============================================================
# STEP 5 OUTPUT
# ============================================================

MODEL_COMPARISON_FILE = (
    OUTPUTS_DIR / "safety_model_comparison.csv"
)

# Name expected by Step 6
SAFETY_COMPARISON_FILE = (
    OUTPUTS_DIR / "safety_model_comparison.csv"
)

# Backward-compatible name
SAFETY_COMBINED_FILE = (
    OUTPUTS_DIR / "safety_model_comparison.csv"
)


# ============================================================
# STEP 6 OUTPUT
# ============================================================

SAFETY_SCORES_FILE = (
    OUTPUTS_DIR / "safety_scores.csv"
)


# ============================================================
# STEP 7 OUTPUTS
# ============================================================

REWARD_FILE = (
    OUTPUTS_DIR / "reward_ranking.csv"
)

RECOMMENDED_FILE = (
    OUTPUTS_DIR / "recommended_shortlist.csv"
)


# ============================================================
# OTHER OUTPUTS
# ============================================================

RUN_SUMMARY_FILE = (
    OUTPUTS_DIR / "run_summary.json"
)

SENSITIVITY_FILE = (
    OUTPUTS_DIR / "sensitivity_analysis.csv"
)


# ============================================================
# TRAINED DNN MODEL
# ============================================================

DNN_MODEL_FILE = (
    MODELS_DIR / "multitask_safety_dnn.pt"
)

DNN_SCALER_FILE = (
    MODELS_DIR / "multitask_safety_scaler.joblib"
)

DNN_CONFIG_FILE = (
    MODELS_DIR / "multitask_safety_dnn_config.json"
)


# ============================================================
# CACHE DIRECTORIES
# ============================================================

TDC_CACHE_DIR = (
    DATA_DIR / "tdc"
)

ADMET_AI_CACHE_DIR = (
    DATA_DIR / "admet_ai"
)


# ============================================================
# SEED FUNCTION
# ============================================================

def seed_everything(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    try:

        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(seed)

        torch.use_deterministic_algorithms(False)

    except ImportError:

        pass


# ============================================================
# DIRECTORY CREATION
# ============================================================

def dirs():

    directories = [
        DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        STRUCTURES_DIR,
        DOCS_DIR,
        RUNS_DIR,
        CODE_DIR,
        TESTS_DIR,
        TDC_CACHE_DIR,
        ADMET_AI_CACHE_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CONFIG VALIDATION
# ============================================================

def check():

    # Dataset split
    if abs(
        TRAIN_FRAC
        + VAL_FRAC
        + TEST_FRAC
        - 1.0
    ) > 1e-9:

        raise ValueError(
            "Split fractions must sum to 1."
        )

    # Reward weights
    if abs(
        W_AFFINITY
        + W_SAFETY
        - 1.0
    ) > 1e-9:

        raise ValueError(
            "Reward weights must sum to 1."
        )

    # Safety weights
    safety_weight_sum = (
        W_QED
        + W_LIPINSKI
        + W_ADME
        + W_TOX
        + W_ALERT
    )

    if abs(
        safety_weight_sum - 1.0
    ) > 1e-9:

        raise ValueError(
            "Safety score weights must sum to 1."
        )

    # Hard thresholds
    if not (
        0 <= AMES_HARD_FAIL_THRESHOLD <= 1
    ):

        raise ValueError(
            "AMES hard-fail threshold must be [0,1]."
        )

    if not (
        0 <= HERG_HARD_FAIL_THRESHOLD <= 1
    ):

        raise ValueError(
            "hERG hard-fail threshold must be [0,1]."
        )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    dirs()

    check()

    print("=" * 70)
    print("MODULE 3 CONFIGURATION")
    print("=" * 70)

    print(
        "Project root:",
        PROJECT_ROOT
    )

    print(
        "Safety comparison:",
        SAFETY_COMPARISON_FILE
    )

    print(
        "Safety scores:",
        SAFETY_SCORES_FILE
    )

    print(
        "DNN model:",
        DNN_MODEL_FILE
    )

    print(
        "DNN scaler:",
        DNN_SCALER_FILE
    )

    print(
        "DNN config:",
        DNN_CONFIG_FILE
    )

    print(
        "Seed:",
        SEED
    )

    print(
        "Configuration: OK"
    )

    print("=" * 70)