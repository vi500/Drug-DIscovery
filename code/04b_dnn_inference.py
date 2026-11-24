from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "outputs" / "validated_candidates.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "multitask_safety_dnn.pt"
SCALER_FILE = PROJECT_ROOT / "models" / "multitask_safety_scaler.joblib"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "dnn_candidate_predictions.csv"


# ============================================================
# SETTINGS
# ============================================================

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

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

CLASSIFICATION_ENDPOINTS = (
    "AMES",
    "hERG",
    "DILI",
    "ClinTox",
    "Carcinogens",
    "BBB",
    "HIA",
)


# ============================================================
# DNN MODEL
# ============================================================

class MultiTaskSafetyDNN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_1,
        hidden_2,
        dropout,
        endpoints,
    ):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_1),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(hidden_1, hidden_2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_2),
            torch.nn.Dropout(dropout),
        )

        self.heads = torch.nn.ModuleDict()

        for endpoint in endpoints:
            self.heads[endpoint] = torch.nn.Linear(
                hidden_2,
                1
            )

    def forward(self, x):

        representation = self.encoder(x)

        outputs = {}

        for endpoint, head in self.heads.items():
            outputs[endpoint] = head(
                representation
            ).squeeze(-1)

        return outputs


# ============================================================
# ERROR HANDLER
# ============================================================

def fail(message):

    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


# ============================================================
# DESCRIPTORS
# ============================================================

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


# ============================================================
# FEATURE GENERATION
# ============================================================

def generate_features(smiles_list):

    generator = GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS
    )

    fingerprints = []
    descriptors = []

    for smiles in smiles_list:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            fail(
                f"Invalid SMILES encountered: {smiles}"
            )

        fingerprint = generator.GetFingerprintAsNumPy(
            mol
        )

        fingerprints.append(
            fingerprint.astype(np.float32)
        )

        descriptors.append(
            calculate_descriptors(mol)
        )

    X_morgan = np.asarray(
        fingerprints,
        dtype=np.float32
    )

    X_desc = np.asarray(
        descriptors,
        dtype=np.float64
    )

    return X_morgan, X_desc


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODULE 3 - DNN CANDIDATE INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        fail(
            f"Input file not found:\n{INPUT_FILE}"
        )

    if not MODEL_FILE.exists():
        fail(
            f"Trained model not found:\n{MODEL_FILE}"
        )

    if not SCALER_FILE.exists():
        fail(
            f"Trained scaler not found:\n{SCALER_FILE}"
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading trained model...")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        fail("Invalid DNN checkpoint.")

    required_keys = [
        "model_state_dict",
        "input_dim",
        "endpoints",
        "hidden_1",
        "hidden_2",
        "dropout",
    ]

    missing = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing:
        fail(
            f"Model checkpoint missing keys: {missing}"
        )

    input_dim = int(
        checkpoint["input_dim"]
    )

    endpoints = tuple(
        checkpoint["endpoints"]
    )

    hidden_1 = int(
        checkpoint["hidden_1"]
    )

    hidden_2 = int(
        checkpoint["hidden_2"]
    )

    dropout = float(
        checkpoint["dropout"]
    )

    print("Model input dimension:", input_dim)
    print("Endpoints:", endpoints)

    # --------------------------------------------------------
    # LOAD TRAINED SCALER
    # --------------------------------------------------------

    print("\nLoading trained scaler...")

    scaler = joblib.load(
        SCALER_FILE
    )

    print(
        "Scaler features:",
        scaler.n_features_in_
    )

    if scaler.n_features_in_ != 8:
        fail(
            "Expected trained scaler with 8 features, "
            f"but found {scaler.n_features_in_}."
        )

    # --------------------------------------------------------
    # LOAD CANDIDATES
    # --------------------------------------------------------

    print("\nLoading candidates...")

    df = pd.read_csv(
        INPUT_FILE
    )

    if "candidate_id" not in df.columns:
        fail(
            "Input file must contain candidate_id."
        )

    if "smiles" not in df.columns:
        fail(
            "Input file must contain smiles."
        )

    if df.empty:
        fail(
            "No candidates found."
        )

    candidate_ids = (
        df["candidate_id"]
        .astype(str)
        .tolist()
    )

    smiles_list = (
        df["smiles"]
        .astype(str)
        .tolist()
    )

    print(
        "Candidates loaded:",
        len(df)
    )

    # --------------------------------------------------------
    # GENERATE FEATURES
    # --------------------------------------------------------

    print("\nGenerating features...")

    X_morgan, X_desc = generate_features(
        smiles_list
    )

    print(
        "Morgan features:",
        X_morgan.shape
    )

    print(
        "RDKit descriptors:",
        X_desc.shape
    )

    # --------------------------------------------------------
    # APPLY TRAINED SCALER
    #
    # IMPORTANT:
    # The saved scaler was fitted on ONLY 8 RDKit descriptors.
    #
    # Morgan = 2048 features -> unchanged
    # RDKit  = 8 features    -> scaled
    # Final  = 2056 features
    # --------------------------------------------------------

    print("\nApplying trained scaler...")

    X_desc_scaled = scaler.transform(
        X_desc
    )

    X_desc_scaled = np.asarray(
        X_desc_scaled,
        dtype=np.float32
    )

    print(
        "Scaled descriptors:",
        X_desc_scaled.shape
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    X_scaled = np.concatenate(
        [
            X_morgan,
            X_desc_scaled
        ],
        axis=1
    ).astype(
        np.float32
    )

    print(
        "Final model input:",
        X_scaled.shape
    )

    # --------------------------------------------------------
    # VERIFY DIMENSION
    # --------------------------------------------------------

    if X_scaled.shape[1] != input_dim:

        fail(
            "Feature dimension mismatch.\n"
            f"Generated: {X_scaled.shape[1]}\n"
            f"Model expects: {input_dim}"
        )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\nBuilding DNN...")

    model = MultiTaskSafetyDNN(
        input_dim=input_dim,
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        dropout=dropout,
        endpoints=endpoints,
    )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Trained DNN loaded successfully."
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print("\nRunning inference...")

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = model(
            X_tensor
        )

    # --------------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["candidate_id"] = candidate_ids
    result["smiles"] = smiles_list

    for endpoint in endpoints:

        raw_output = (
            outputs[endpoint]
            .cpu()
            .numpy()
        )

        if endpoint in CLASSIFICATION_ENDPOINTS:

            probability = (
                1.0 /
                (
                    1.0 +
                    np.exp(
                        -np.clip(
                            raw_output,
                            -50,
                            50
                        )
                    )
                )
            )

            result[endpoint] = probability

        else:

            result[endpoint] = raw_output

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DNN INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "Candidates:",
        len(result)
    )

    print(
        "Final features:",
        X_scaled.shape[1]
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print("\nPredictions:")
    print(
        result.to_string(
            index=False
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    main()