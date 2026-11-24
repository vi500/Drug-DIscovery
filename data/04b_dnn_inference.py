from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


MODEL_PATH = MODELS_DIR / "multitask_safety_dnn.pt"
SCALER_PATH = MODELS_DIR / "multitask_safety_scaler.joblib"


# ---------------------------------------------------------------------
# Same architecture used during training
# ---------------------------------------------------------------------
class MultiTaskDNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        endpoints: list[str],
        hidden_1: int,
        hidden_2: int,
        dropout: float,
    ):
        super().__init__()

        self.endpoints = endpoints

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict()

        for endpoint in endpoints:
            if endpoint == "Solubility":
                self.heads[endpoint] = nn.Linear(hidden_2, 1)
            else:
                self.heads[endpoint] = nn.Linear(hidden_2, 1)

    def forward(self, x):
        z = self.encoder(x)

        outputs = {}

        for endpoint in self.endpoints:
            outputs[endpoint] = self.heads[endpoint](z).squeeze(1)

        return outputs


# ---------------------------------------------------------------------
# RDKit descriptors - SAME descriptors used during training
# ---------------------------------------------------------------------
DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "TPSA",
    "MolMR",
    "FractionCSP3",
)


def calculate_descriptors(mol):
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]


def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None, None

    canonical = Chem.MolToSmiles(mol, canonical=True)

    return canonical, mol


def build_features(smiles_list, radius, n_bits):
    """
    Build exactly:

        Morgan fingerprint = 2048
        RDKit descriptors   = 8

        Total               = 2056
    """

    fingerprints = []
    descriptors = []
    canonical_smiles = []

    for smiles in smiles_list:

        canonical, mol = canonicalize_smiles(smiles)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES encountered during inference: {smiles}"
            )

        generator = AllChem.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )

        fp = generator.GetFingerprintAsNumPy(mol)

        desc = calculate_descriptors(mol)

        fingerprints.append(fp)
        descriptors.append(desc)
        canonical_smiles.append(canonical)

    X_fp = np.asarray(fingerprints, dtype=np.float32)
    X_desc = np.asarray(descriptors, dtype=np.float32)

    X = np.concatenate([X_fp, X_desc], axis=1)

    return X, canonical_smiles


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Module 3 DNN inference"
    )

    parser.add_argument(
        "--input",
        default=str(OUTPUTS_DIR / "validated_candidates.csv"),
        help="Input candidate CSV",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "dnn_candidate_predictions.csv"),
        help="Output prediction CSV",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("MODULE 3 - MULTI-ENDPOINT DNN INFERENCE")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Hard-fail checks
    # -----------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Required trained model not found:\n{MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Required scaler not found:\n{SCALER_PATH}"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input candidate file not found:\n{input_path}"
        )

    print(f"Model  : {MODEL_PATH}")
    print(f"Scaler : {SCALER_PATH}")
    print(f"Input  : {input_path}")

    # -----------------------------------------------------------------
    # Load model checkpoint
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "input_dim",
        "endpoints",
        "hidden_1",
        "hidden_2",
        "dropout",
        "morgan_radius",
        "morgan_bits",
        "use_rdkit_descriptors",
        "descriptor_names",
        "seed",
    }

    missing = required_keys.difference(checkpoint.keys())

    if missing:
        raise RuntimeError(
            f"Model checkpoint is missing required keys: {sorted(missing)}"
        )

    endpoints = list(checkpoint["endpoints"])

    print("\nModel configuration:")
    print(f"  Input dimension : {checkpoint['input_dim']}")
    print(f"  Endpoints       : {endpoints}")
    print(f"  Morgan radius   : {checkpoint['morgan_radius']}")
    print(f"  Morgan bits     : {checkpoint['morgan_bits']}")
    print(f"  RDKit features  : {checkpoint['use_rdkit_descriptors']}")
    print(f"  Seed            : {checkpoint['seed']}")

    # -----------------------------------------------------------------
    # Load scaler
    # -----------------------------------------------------------------

    scaler = joblib.load(SCALER_PATH)

    # -----------------------------------------------------------------
    # Load candidates
    # -----------------------------------------------------------------

    df = pd.read_csv(input_path)

    required_columns = {"candidate_id", "smiles"}

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input file missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["candidate_id"].duplicated().any():
        duplicates = df.loc[
            df["candidate_id"].duplicated(),
            "candidate_id"
        ].tolist()

        raise ValueError(
            f"Duplicate candidate_id values detected: {duplicates[:10]}"
        )

    if df["smiles"].isna().any():
        raise ValueError("Input contains missing SMILES values.")

    print(f"\nCandidates loaded: {len(df)}")

    # -----------------------------------------------------------------
    # Generate features
    # -----------------------------------------------------------------

    X, canonical_smiles = build_features(
        df["smiles"].tolist(),
        radius=int(checkpoint["morgan_radius"]),
        n_bits=int(checkpoint["morgan_bits"]),
    )

    print(f"Morgan + descriptors shape: {X.shape}")

    if X.shape[1] != int(checkpoint["input_dim"]):
        raise RuntimeError(
            "Feature dimension mismatch.\n"
            f"Expected: {checkpoint['input_dim']}\n"
            f"Actual:   {X.shape[1]}"
        )

    # -----------------------------------------------------------------
    # Apply trained scaler
    # -----------------------------------------------------------------

    X_scaled = scaler.transform(X).astype(np.float32)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    # -----------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------

    model = MultiTaskDNN(
        input_dim=int(checkpoint["input_dim"]),
        endpoints=endpoints,
        hidden_1=int(checkpoint["hidden_1"]),
        hidden_2=int(checkpoint["hidden_2"]),
        dropout=float(checkpoint["dropout"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("Model weights loaded successfully.")

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    with torch.no_grad():

        raw_predictions = model(X_tensor)

    # -----------------------------------------------------------------
    # Create output
    # -----------------------------------------------------------------

    result = df[
        ["candidate_id", "smiles"]
    ].copy()

    result["canonical_smiles"] = canonical_smiles

    for endpoint in endpoints:

        values = raw_predictions[endpoint].cpu().numpy()

        if endpoint == "Solubility":
            # Regression output: retain real predicted value.
            result[endpoint] = values

        else:
            # Classification output: convert logits to probability.
            probabilities = 1.0 / (1.0 + np.exp(-values))

            result[f"{endpoint}_prob"] = probabilities

            result[f"{endpoint}_pred"] = (
                probabilities >= 0.50
            ).astype(int)

    # -----------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------

    probability_columns = [
        f"{endpoint}_prob"
        for endpoint in endpoints
        if endpoint != "Solubility"
    ]

    for column in probability_columns:

        if result[column].isna().any():
            raise RuntimeError(
                f"NaN predictions detected in {column}"
            )

        if not (
            (result[column] >= 0.0)
            & (result[column] <= 1.0)
        ).all():

            raise RuntimeError(
                f"Probability outside [0,1] detected in {column}"
            )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DNN INFERENCE COMPLETE")
    print("=" * 70)

    print(f"Rows written : {len(result)}")
    print(f"Columns      : {len(result.columns)}")
    print(f"Output       : {output_path}")

    print("\nPrediction columns:")

    for column in result.columns:
        print(f"  {column}")

    print("\nFirst predictions:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()