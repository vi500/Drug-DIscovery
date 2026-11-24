# ================================================================
# MODULE 3 - MULTI-ENDPOINT DRUG SAFETY DNN
# ================================================================
# CPU-only
# Python 3.13 compatible
# Uses locally downloaded official TDC datasets.
#
# Endpoints:
#   AMES
#   hERG
#   DILI
#   ClinTox
#   Carcinogens
#   BBB
#   HIA
#   Solubility
#
# Architecture:
#   SMILES
#      |
#      +--> Morgan fingerprint
#      |
#      +--> RDKit descriptors
#      |
#      v
#   Shared neural network encoder
#      |
#      +--> AMES head
#      +--> hERG head
#      +--> DILI head
#      +--> ClinTox head
#      +--> Carcinogens head
#      +--> BBB head
#      +--> HIA head
#      +--> Solubility head
#
# Missing labels are masked during training.
# ================================================================

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(r"D:\Major project\safety_module")

DATA_DIR = PROJECT_ROOT / "data"
TDC_DIR = DATA_DIR / "tdc"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

ENDPOINTS = [
    "AMES",
    "hERG",
    "DILI",
    "ClinTox",
    "Carcinogens",
    "BBB",
    "HIA",
    "Solubility",
]

CLASSIFICATION_ENDPOINTS = [
    "AMES",
    "hERG",
    "DILI",
    "ClinTox",
    "Carcinogens",
    "BBB",
    "HIA",
]

REGRESSION_ENDPOINTS = [
    "Solubility",
]


# Local TDC files downloaded earlier.
LOCAL_DATASETS = {
    "AMES": TDC_DIR / "ames_download",
    "hERG": TDC_DIR / "herg_download",
    "DILI": TDC_DIR / "dili_download",
    "ClinTox": TDC_DIR / "clintox_download",
    "Carcinogens": TDC_DIR / "carcinogens_download",
    "BBB": TDC_DIR / "bbb_download",
    "HIA": TDC_DIR / "hia_download",
    "Solubility": TDC_DIR / "solubility_download",
}


# Model configuration
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

USE_RDKIT_DESCRIPTORS = True

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "MolMR",
    "FractionCSP3",
]

HIDDEN_1 = 512
HIDDEN_2 = 256

DROPOUT = 0.30

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 12

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CPU-only deterministic behaviour.
    torch.use_deterministic_algorithms(False)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device("cpu")


# ================================================================
# ENDPOINT INFORMATION
# ================================================================

def is_classification(endpoint: str) -> bool:
    return endpoint in CLASSIFICATION_ENDPOINTS


# ================================================================
# SMILES CLEANING
# ================================================================

def clean_smiles(value) -> str:
    """
    Clean accidental surrounding whitespace/quotes.

    This is particularly important for the BBB TDC file, where
    malformed quoting can otherwise become part of the SMILES.
    """

    if pd.isna(value):
        return ""

    s = str(value)

    s = s.strip()

    # Remove surrounding double quotes.
    s = s.strip('"')

    # Remove surrounding single quotes.
    s = s.strip("'")

    s = s.strip()

    return s


# ================================================================
# DATASET READER
# ================================================================

def read_tdc_file(path: Path) -> pd.DataFrame:
    """
    Read a locally downloaded TDC dataset.

    TDC files are normally tab-separated, but some files contain
    inconsistent quoting. Therefore the reader first attempts the
    normal parser and then uses a controlled fallback parser.

    No rows are silently dropped.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"TDC dataset not found:\n{path}"
        )

    # ------------------------------------------------------------
    # Attempt 1: normal tab-separated parser
    # ------------------------------------------------------------

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            engine="python",
        )

    except Exception as first_error:

        print(
            f"    Standard parser failed: {type(first_error).__name__}"
        )

        # --------------------------------------------------------
        # Attempt 2: quote characters treated literally
        # --------------------------------------------------------

        try:
            df = pd.read_csv(
                path,
                sep="\t",
                engine="python",
                quoting=3,
            )

        except Exception as second_error:

            raise RuntimeError(
                f"Could not parse TDC dataset:\n"
                f"{path}\n\n"
                f"Standard parser error:\n"
                f"{first_error}\n\n"
                f"Literal-quote parser error:\n"
                f"{second_error}"
            ) from second_error

    return df


# ================================================================
# LOCAL TDC LOADER
# ================================================================

def load_local_tdc(endpoint: str) -> pd.DataFrame:
    """
    Load one locally cached TDC dataset.

    Expected columns:
        Drug
        Y

    Returned columns:
        smiles
        <endpoint>
    """

    if endpoint not in LOCAL_DATASETS:
        raise KeyError(
            f"No local dataset mapping exists for {endpoint}"
        )

    path = LOCAL_DATASETS[endpoint]

    print(f"  Loading {endpoint}...")

    df = read_tdc_file(path)

    print(
        f"    Raw rows      : {len(df)}"
    )

    print(
        f"    Raw columns   : {list(df.columns)}"
    )

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------

    if "Drug" not in df.columns:
        raise RuntimeError(
            f"{endpoint} dataset does not contain 'Drug' column.\n"
            f"Available columns: {list(df.columns)}\n"
            f"File: {path}"
        )

    if "Y" not in df.columns:
        raise RuntimeError(
            f"{endpoint} dataset does not contain 'Y' column.\n"
            f"Available columns: {list(df.columns)}\n"
            f"File: {path}"
        )

    # ------------------------------------------------------------
    # Clean SMILES
    # ------------------------------------------------------------

    df["Drug"] = (
        df["Drug"]
        .astype(str)
        .map(clean_smiles)
    )

    # ------------------------------------------------------------
    # Validate molecules
    # ------------------------------------------------------------

    valid_smiles = []
    valid_y = []

    invalid_count = 0

    for smiles, y in zip(
        df["Drug"].tolist(),
        df["Y"].tolist(),
    ):

        if not smiles:
            invalid_count += 1
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            invalid_count += 1
            continue

        # Canonicalize SMILES.
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
        )

        if not canonical:
            invalid_count += 1
            continue

        valid_smiles.append(canonical)
        valid_y.append(y)

    if len(valid_smiles) == 0:
        raise RuntimeError(
            f"No valid molecules remained for endpoint {endpoint}.\n"
            f"File: {path}"
        )

    result = pd.DataFrame(
        {
            "smiles": valid_smiles,
            endpoint: valid_y,
        }
    )

    # ------------------------------------------------------------
    # Convert labels to numeric
    # ------------------------------------------------------------

    result[endpoint] = pd.to_numeric(
        result[endpoint],
        errors="coerce",
    )

    # Keep missing labels for multitask masking.
    # Only remove rows where SMILES is invalid.
    # Missing Y values remain NaN.

    # ------------------------------------------------------------
    # Remove duplicate SMILES
    # ------------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            subset=["smiles"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"    Valid molecules: {len(result)}"
    )

    print(
        f"    Invalid SMILES : {invalid_count}"
    )

    return result


# ================================================================
# MORGAN FINGERPRINT
# ================================================================

def make_morgan_fingerprint(smiles: str) -> np.ndarray:
    """
    Generate a Morgan fingerprint using the modern RDKit API.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES encountered during fingerprinting: {smiles}"
        )

    fp = AllChem.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    ).GetFingerprint(mol)

    arr = np.zeros(
        (MORGAN_BITS,),
        dtype=np.float32,
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr,
    )

    return arr


# ================================================================
# RDKIT DESCRIPTORS
# ================================================================

def calculate_descriptors(smiles: str) -> np.ndarray:
    """
    Calculate optional RDKit molecular descriptors.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    values = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    return values


# ================================================================
# FEATURE GENERATION
# ================================================================

def build_features(
    smiles_list: list[str],
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build Morgan fingerprints and RDKit descriptors.
    """

    fingerprints = []
    descriptors = []

    print(
        f"  Generating features for {len(smiles_list)} molecules..."
    )

    for i, smiles in enumerate(smiles_list):

        fingerprints.append(
            make_morgan_fingerprint(smiles)
        )

        if USE_RDKIT_DESCRIPTORS:
            descriptors.append(
                calculate_descriptors(smiles)
            )

        if (i + 1) % 500 == 0:
            print(
                f"    Features: {i + 1}/{len(smiles_list)}"
            )

    X_fp = np.asarray(
        fingerprints,
        dtype=np.float32,
    )

    if USE_RDKIT_DESCRIPTORS:

        X_desc = np.asarray(
            descriptors,
            dtype=np.float32,
        )

    else:

        X_desc = None

    return X_fp, X_desc


# ================================================================
# MULTITASK DATASET MERGING
# ================================================================

def merge_endpoint_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge endpoint datasets by canonical SMILES.

    Missing labels are preserved as NaN and later masked.
    """

    merged = None

    for endpoint in ENDPOINTS:

        df = datasets[endpoint][
            ["smiles", endpoint]
        ].copy()

        if merged is None:

            merged = df

        else:

            merged = merged.merge(
                df,
                on="smiles",
                how="outer",
                validate="one_to_one",
            )

    if merged is None or len(merged) == 0:
        raise RuntimeError(
            "Multitask dataset is empty."
        )

    merged = (
        merged
        .drop_duplicates(
            subset=["smiles"]
        )
        .reset_index(drop=True)
    )

    return merged


# ================================================================
# SCAFFOLD
# ================================================================

def get_scaffold(smiles: str) -> str:
    """
    Generate Bemis-Murcko scaffold.
    """

    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES for scaffold: {smiles}"
        )

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )


def scaffold_split(
    smiles: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaffold-based train/validation/test split.
    """

    rng = random.Random(seed)

    scaffold_to_indices: dict[str, list[int]] = {}

    for idx, smiles_value in enumerate(smiles):

        scaffold = get_scaffold(smiles_value)

        scaffold_to_indices.setdefault(
            scaffold,
            [],
        ).append(idx)

    scaffolds = list(
        scaffold_to_indices.keys()
    )

    rng.shuffle(scaffolds)

    # Sort by group size after shuffle.
    scaffolds.sort(
        key=lambda x: len(scaffold_to_indices[x]),
        reverse=True,
    )

    n_total = len(smiles)

    train_target = int(
        TRAIN_FRAC * n_total
    )

    val_target = int(
        VAL_FRAC * n_total
    )

    train_idx = []
    val_idx = []
    test_idx = []

    for scaffold in scaffolds:

        group = scaffold_to_indices[scaffold]

        if len(train_idx) + len(group) <= train_target:

            train_idx.extend(group)

        elif len(val_idx) + len(group) <= val_target:

            val_idx.extend(group)

        else:

            test_idx.extend(group)

    # If validation or test became empty, redistribute carefully.
    if len(val_idx) == 0 or len(test_idx) == 0:

        raise RuntimeError(
            "Scaffold split produced an empty validation/test set.\n"
            f"Train={len(train_idx)}, "
            f"Val={len(val_idx)}, "
            f"Test={len(test_idx)}"
        )

    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


# ================================================================
# MULTITASK DNN
# ================================================================

class MultiTaskDNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        endpoints: list[str],
    ):

        super().__init__()

        self.endpoints = endpoints

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_1,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                HIDDEN_1
            ),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                HIDDEN_2
            ),

            nn.Dropout(
                DROPOUT
            ),
        )

        self.heads = nn.ModuleDict()

        for endpoint in endpoints:

            self.heads[endpoint] = nn.Linear(
                HIDDEN_2,
                1,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        representation = self.encoder(x)

        outputs = {}

        for endpoint in self.endpoints:

            outputs[endpoint] = self.heads[
                endpoint
            ](
                representation
            ).squeeze(-1)

        return outputs


# ================================================================
# LOSS
# ================================================================

def calculate_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    endpoint_index: dict[str, int],
) -> tuple[torch.Tensor, dict[str, float]]:

    losses = {}
    total = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=DEVICE,
    )

    used = 0

    for endpoint in ENDPOINTS:

        idx = endpoint_index[endpoint]

        target = targets[:, idx]

        mask = ~torch.isnan(target)

        if mask.sum().item() == 0:
            continue

        prediction = outputs[endpoint]

        if is_classification(endpoint):

            loss_function = nn.BCEWithLogitsLoss()

            endpoint_loss = loss_function(
                prediction[mask],
                target[mask],
            )

        else:

            loss_function = nn.MSELoss()

            endpoint_loss = loss_function(
                prediction[mask],
                target[mask],
            )

        total = total + endpoint_loss

        losses[endpoint] = float(
            endpoint_loss.detach().cpu().item()
        )

        used += 1

    if used == 0:

        raise RuntimeError(
            "No valid labels were available for this batch."
        )

    total = total / used

    return total, losses


# ================================================================
# PREDICTIONS
# ================================================================

def convert_outputs_to_predictions(
    outputs: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:

    predictions = {}

    for endpoint in ENDPOINTS:

        values = (
            outputs[endpoint]
            .detach()
            .cpu()
            .numpy()
        )

        if is_classification(endpoint):

            values = 1.0 / (
                1.0 + np.exp(-values)
            )

        predictions[endpoint] = values

    return predictions


# ================================================================
# EVALUATION
# ================================================================

def evaluate_model(
    model: MultiTaskDNN,
    X: np.ndarray,
    Y: np.ndarray,
    indices: np.ndarray,
) -> dict:

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X[indices],
            dtype=torch.float32,
            device=DEVICE,
        )

        outputs = model(
            X_tensor
        )

    predictions = convert_outputs_to_predictions(
        outputs
    )

    metrics = {}

    for endpoint in ENDPOINTS:

        endpoint_idx = ENDPOINTS.index(
            endpoint
        )

        y_true = Y[
            indices,
            endpoint_idx
        ]

        y_pred = predictions[
            endpoint
        ]

        mask = ~np.isnan(y_true)

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        metric_row = {
            "endpoint": endpoint,
            "task": (
                "classification"
                if is_classification(endpoint)
                else "regression"
            ),
            "n": int(len(y_true)),
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "Accuracy": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
        }

        if len(y_true) == 0:
            metrics[endpoint] = metric_row
            continue

        if is_classification(endpoint):

            unique_values = np.unique(
                y_true
            )

            if len(unique_values) >= 2:

                metric_row["AUROC"] = float(
                    roc_auc_score(
                        y_true,
                        y_pred,
                    )
                )

                metric_row["AUPRC"] = float(
                    average_precision_score(
                        y_true,
                        y_pred,
                    )
                )

            binary_pred = (
                y_pred >= 0.5
            ).astype(int)

            metric_row["Accuracy"] = float(
                accuracy_score(
                    y_true,
                    binary_pred,
                )
            )

        else:

            rmse = np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )

            metric_row["RMSE"] = float(
                rmse
            )

            if len(y_true) >= 2:

                try:

                    metric_row["R2"] = float(
                        r2_score(
                            y_true,
                            y_pred,
                        )
                    )

                except Exception:

                    metric_row["R2"] = np.nan

        metrics[endpoint] = metric_row

    return metrics


# ================================================================
# TRAINING
# ================================================================

def train_model(
    model: MultiTaskDNN,
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
) -> tuple[MultiTaskDNN, list[dict]]:

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    endpoint_index = {
        endpoint: idx
        for idx, endpoint in enumerate(
            ENDPOINTS
        )
    }

    X_train = torch.tensor(
        X[train_idx],
        dtype=torch.float32,
    )

    Y_train = torch.tensor(
        Y[train_idx],
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_train,
        Y_train,
    )

    generator = torch.Generator()

    generator.manual_seed(
        torch.initial_seed()
    )

    loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    generator=generator,
)

    best_score = float("inf")
    best_state = None
    patience_counter = 0

    history = []

    print()
    print(
        "  Training multitask DNN..."
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        epoch_losses = []

        for batch_X, batch_Y in loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_Y = batch_Y.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                batch_X
            )

            loss, _ = calculate_loss(
                outputs,
                batch_Y,
                endpoint_index,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach()
                    .cpu()
                    .item()
                )
            )

        train_loss = float(
            np.mean(epoch_losses)
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_X = torch.tensor(
                X[val_idx],
                dtype=torch.float32,
                device=DEVICE,
            )

            val_Y = torch.tensor(
                Y[val_idx],
                dtype=torch.float32,
                device=DEVICE,
            )

            val_outputs = model(
                val_X
            )

            val_loss, _ = calculate_loss(
                val_outputs,
                val_Y,
                endpoint_index,
            )

            val_loss_value = float(
                val_loss
                .detach()
                .cpu()
                .item()
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_value,
            }
        )

        print(
            f"    Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss_value:.6f}"
        )

        # --------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------

        if val_loss_value < best_score:

            best_score = val_loss_value

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(
                    f"    Early stopping at epoch {epoch}"
                )

                break

    if best_state is None:

        raise RuntimeError(
            "Training completed without a valid best model."
        )

    model.load_state_dict(
        best_state
    )

    return model, history


# ================================================================
# SAVE MODEL
# ================================================================

def save_model(
    model: MultiTaskDNN,
    input_dim: int,
    scaler: StandardScaler,
    seed: int,
    history: list[dict],
) -> None:

    model_path = (
        MODELS_DIR /
        "multitask_safety_dnn.pt"
    )

    scaler_path = (
        MODELS_DIR /
        "multitask_safety_scaler.joblib"
    )

    config_path = (
        MODELS_DIR /
        "multitask_safety_dnn_config.json"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                input_dim,

            "endpoints":
                ENDPOINTS,

            "hidden_1":
                HIDDEN_1,

            "hidden_2":
                HIDDEN_2,

            "dropout":
                DROPOUT,

            "morgan_radius":
                MORGAN_RADIUS,

            "morgan_bits":
                MORGAN_BITS,

            "use_rdkit_descriptors":
                USE_RDKIT_DESCRIPTORS,

            "descriptor_names":
                DESCRIPTOR_NAMES,

            "seed":
                seed,
        },
        model_path,
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "input_dim": input_dim,
                "endpoints": ENDPOINTS,
                "hidden_1": HIDDEN_1,
                "hidden_2": HIDDEN_2,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "morgan_radius": MORGAN_RADIUS,
                "morgan_bits": MORGAN_BITS,
                "use_rdkit_descriptors":
                    USE_RDKIT_DESCRIPTORS,
                "descriptor_names":
                    DESCRIPTOR_NAMES,
                "seed": seed,
                "history": history,
            },
            f,
            indent=2,
        )

    print()
    print(
        f"  Model saved  : {model_path}"
    )

    print(
        f"  Scaler saved : {scaler_path}"
    )

    print(
        f"  Config saved : {config_path}"
    )


# ================================================================
# SAVE METRICS
# ================================================================

def save_metrics(
    test_metrics: dict,
    val_metrics: dict,
    history: list[dict],
) -> None:

    rows = []

    for endpoint in ENDPOINTS:

        row = test_metrics[endpoint].copy()

        row["split"] = "test"

        rows.append(row)

    for endpoint in ENDPOINTS:

        row = val_metrics[endpoint].copy()

        row["split"] = "validation"

        rows.append(row)

    metrics_df = pd.DataFrame(
        rows
    )

    metrics_path = (
        OUTPUTS_DIR /
        "dnn_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    history_path = (
        OUTPUTS_DIR /
        "dnn_training_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    print()
    print(
        f"  Metrics saved  : {metrics_path}"
    )

    print(
        f"  History saved  : {history_path}"
    )


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

def save_test_predictions(
    model: MultiTaskDNN,
    df: pd.DataFrame,
    X: np.ndarray,
    test_idx: np.ndarray,
) -> None:

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X[test_idx],
            dtype=torch.float32,
            device=DEVICE,
        )

        outputs = model(
            X_tensor
        )

    predictions = convert_outputs_to_predictions(
        outputs
    )

    prediction_df = pd.DataFrame(
        {
            "smiles":
                df.iloc[test_idx]["smiles"].values
        }
    )

    for endpoint in ENDPOINTS:

        prediction_df[
            f"{endpoint}_prediction"
        ] = predictions[
            endpoint
        ]

    # Also save real labels.
    for endpoint in ENDPOINTS:

        prediction_df[
            f"{endpoint}_actual"
        ] = df.iloc[
            test_idx
        ][endpoint].values

    prediction_path = (
        OUTPUTS_DIR /
        "dnn_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"  Predictions saved: {prediction_path}"
    )


# ================================================================
# MODEL SUMMARY
# ================================================================

def print_metrics(
    metrics: dict,
    title: str,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for endpoint in ENDPOINTS:

        row = metrics[endpoint]

        if row["task"] == "classification":

            auroc = row["AUROC"]

            auprc = row["AUPRC"]

            accuracy = row["Accuracy"]

            auroc_text = (
                f"{auroc:.4f}"
                if not np.isnan(auroc)
                else "N/A"
            )

            auprc_text = (
                f"{auprc:.4f}"
                if not np.isnan(auprc)
                else "N/A"
            )

            acc_text = (
                f"{accuracy:.4f}"
                if not np.isnan(accuracy)
                else "N/A"
            )

            print(
                f"{endpoint:15s} | "
                f"N={row['n']:5d} | "
                f"AUROC={auroc_text:>8s} | "
                f"AUPRC={auprc_text:>8s} | "
                f"ACC={acc_text:>8s}"
            )

        else:

            rmse = row["RMSE"]

            r2 = row["R2"]

            rmse_text = (
                f"{rmse:.4f}"
                if not np.isnan(rmse)
                else "N/A"
            )

            r2_text = (
                f"{r2:.4f}"
                if not np.isnan(r2)
                else "N/A"
            )

            print(
                f"{endpoint:15s} | "
                f"N={row['n']:5d} | "
                f"RMSE={rmse_text:>8s} | "
                f"R2={r2_text:>8s}"
            )


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Module 3 Multi-Endpoint Drug Safety DNN"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Seed BEFORE model creation
    # ------------------------------------------------------------

    set_seed(
        args.seed
    )

    start_time = time.time()

    print("=" * 70)
    print(
        "MODULE 3 - MULTI-ENDPOINT DRUG SAFETY DNN"
    )
    print("=" * 70)

    print(
        f"Python seed       : {args.seed}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    print(
        f"Endpoints         : {len(ENDPOINTS)}"
    )

    print()

    # ============================================================
    # 1. LOAD TDC DATASETS
    # ============================================================

    print(
        "[1/6] Loading verified TDC datasets..."
    )

    datasets = {}

    for endpoint in ENDPOINTS:

        datasets[endpoint] = load_local_tdc(
            endpoint
        )

    print()
    print(
        "  All TDC datasets loaded successfully."
    )

    # ============================================================
    # 2. MERGE DATASETS
    # ============================================================

    print()
    print(
        "[2/6] Building multitask dataset..."
    )

    df = merge_endpoint_datasets(
        datasets
    )

    print(
        f"  Unique molecules: {len(df)}"
    )

    for endpoint in ENDPOINTS:

        available = (
            df[endpoint]
            .notna()
            .sum()
        )

        print(
            f"  {endpoint:15s}: "
            f"{available} labels"
        )

    # Save merged training dataset.
    merged_path = (
        DATA_DIR /
        "multitask_training_dataset.csv"
    )

    df.to_csv(
        merged_path,
        index=False,
    )

    print(
        f"  Dataset saved: {merged_path}"
    )

    # ============================================================
    # 3. SCAFFOLD SPLIT
    # ============================================================

    print()
    print(
        "[3/6] Creating scaffold split..."
    )

    smiles_list = df[
        "smiles"
    ].tolist()

    train_idx, val_idx, test_idx = (
        scaffold_split(
            smiles_list,
            args.seed,
        )
    )

    print(
        f"  Train: {len(train_idx)}"
    )

    print(
        f"  Validation: {len(val_idx)}"
    )

    print(
        f"  Test: {len(test_idx)}"
    )

    # ============================================================
    # 4. FEATURE GENERATION
    # ============================================================

    print()
    print(
        "[4/6] Generating molecular features..."
    )

    X_fp, X_desc = build_features(
        smiles_list
    )

    if X_desc is not None:

        X = np.concatenate(
            [
                X_fp,
                X_desc,
            ],
            axis=1,
        )

    else:

        X = X_fp

    print(
        f"  Morgan features : {X_fp.shape}"
    )

    if X_desc is not None:

        print(
            f"  RDKit descriptors: {X_desc.shape}"
        )

    print(
        f"  Final feature matrix: {X.shape}"
    )

    # ============================================================
    # SCALE ONLY RDKit DESCRIPTORS
    # ============================================================

    scaler = StandardScaler()

    if USE_RDKIT_DESCRIPTORS:

        descriptor_start = MORGAN_BITS

        scaler.fit(
            X[
                train_idx,
                descriptor_start:
            ]
        )

        X_scaled = X.copy()

        X_scaled[
            :,
            descriptor_start:
        ] = scaler.transform(
            X[
                :,
                descriptor_start:
            ]
        )

        X = X_scaled

    else:

        scaler = StandardScaler()

        # Dummy scaler object for consistent saving.
        scaler.fit(
            np.zeros(
                (
                    len(train_idx),
                    1,
                ),
                dtype=np.float32,
            )
        )

    # ============================================================
    # TARGET MATRIX
    # ============================================================

    Y = np.column_stack(
        [
            pd.to_numeric(
                df[endpoint],
                errors="coerce",
            ).to_numpy(
                dtype=np.float32
            )
            for endpoint in ENDPOINTS
        ]
    )

    # ============================================================
    # 5. TRAIN MODEL
    # ============================================================

    print()
    print(
        "[5/6] Training multi-task DNN..."
    )

    model = MultiTaskDNN(
        input_dim=X.shape[1],
        endpoints=ENDPOINTS,
    ).to(
        DEVICE
    )

    model, history = train_model(
        model=model,
        X=X,
        Y=Y,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # ============================================================
    # VALIDATION METRICS
    # ============================================================

    val_metrics = evaluate_model(
        model,
        X,
        Y,
        val_idx,
    )

    print_metrics(
        val_metrics,
        "VALIDATION METRICS",
    )

    # ============================================================
    # TEST METRICS
    # ============================================================

    test_metrics = evaluate_model(
        model,
        X,
        Y,
        test_idx,
    )

    print_metrics(
        test_metrics,
        "TEST METRICS",
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    save_model(
        model=model,
        input_dim=X.shape[1],
        scaler=scaler,
        seed=args.seed,
        history=history,
    )

    # ============================================================
    # SAVE METRICS
    # ============================================================

    save_metrics(
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        history=history,
    )

    # ============================================================
    # SAVE TEST PREDICTIONS
    # ============================================================

    save_test_predictions(
        model=model,
        df=df,
        X=X,
        test_idx=test_idx,
    )

    # ============================================================
    # SAVE SPLIT INFORMATION
    # ============================================================

    split_df = pd.DataFrame(
        {
            "smiles":
                df["smiles"].values,
            "split":
                [
                    (
                        "train"
                        if i in set(train_idx)
                        else
                        "validation"
                        if i in set(val_idx)
                        else
                        "test"
                    )
                    for i in range(len(df))
                ],
        }
    )

    split_path = (
        OUTPUTS_DIR /
        "dnn_scaffold_split.csv"
    )

    split_df.to_csv(
        split_path,
        index=False,
    )

    print(
        f"  Split saved: {split_path}"
    )

    # ============================================================
    # SAVE RUN INFORMATION
    # ============================================================

    runtime = (
        time.time()
        - start_time
    )

    run_info = {
        "module": "Module 3",
        "model": "MultiTaskDNN",
        "seed": args.seed,
        "device": str(DEVICE),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "total_molecules": len(df),
        "train_molecules": len(train_idx),
        "validation_molecules": len(val_idx),
        "test_molecules": len(test_idx),
        "feature_dimension": int(X.shape[1]),
        "morgan_bits": MORGAN_BITS,
        "morgan_radius": MORGAN_RADIUS,
        "rdkit_descriptors": USE_RDKIT_DESCRIPTORS,
        "endpoints": ENDPOINTS,
        "runtime_seconds": runtime,
    }

    run_info_path = (
        OUTPUTS_DIR /
        "dnn_run_info.json"
    )

    with open(
        run_info_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_info,
            f,
            indent=2,
        )

    # ============================================================
    # FINISH
    # ============================================================

    print()
    print("=" * 70)
    print(
        "MODULE 3 DNN TRAINING COMPLETED"
    )
    print("=" * 70)

    print(
        f"Runtime           : {runtime:.2f} seconds"
    )

    print(
        f"Total molecules   : {len(df)}"
    )

    print(
        f"Train             : {len(train_idx)}"
    )

    print(
        f"Validation        : {len(val_idx)}"
    )

    print(
        f"Test              : {len(test_idx)}"
    )

    print()
    print(
        "Generated files:"
    )

    print(
        "  models\\multitask_safety_dnn.pt"
    )

    print(
        "  models\\multitask_safety_scaler.joblib"
    )

    print(
        "  models\\multitask_safety_dnn_config.json"
    )

    print(
        "  outputs\\dnn_metrics.csv"
    )

    print(
        "  outputs\\dnn_training_history.csv"
    )

    print(
        "  outputs\\dnn_test_predictions.csv"
    )

    print(
        "  outputs\\dnn_scaffold_split.csv"
    )

    print(
        "  outputs\\dnn_run_info.json"
    )

    print("=" * 70)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()