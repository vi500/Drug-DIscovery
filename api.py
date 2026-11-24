# ================================================================
# MODULE 3 - DEEP SAFETY MODEL API
# ================================================================
# Production API
#
# Pipeline:
# TANGen -> EquiBind -> Module 3 -> Stage 5
#
# Model:
# 2048 Morgan fingerprint + 8 RDKit descriptors = 2056 features
#
# IMPORTANT:
# The StandardScaler scales ONLY the 8 RDKit descriptors.
# The 2048 Morgan fingerprint is NOT scaled.
# This exactly matches the training script.
# ================================================================

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    rdMolDescriptors,
)

# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

CODE_DIR = BASE_DIR / "code"
MODELS_DIR = BASE_DIR / "models"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MODEL_PATH = MODELS_DIR / "multitask_safety_dnn.pt"
SCALER_PATH = MODELS_DIR / "multitask_safety_scaler.joblib"

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

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

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

INPUT_DIM = MORGAN_BITS + len(DESCRIPTOR_NAMES)

# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device("cpu")

# ================================================================
# FASTAPI
# ================================================================

app = FastAPI(
    title="Module 3 - Deep Safety Model",
    description=(
        "Hybrid Multi-task DNN + ADMET-AI drug safety, "
        "ADME and drug-likeness screening API."
    ),
    version="1.0.0",
)

# ================================================================
# REQUEST MODELS
# ================================================================

from typing import List
from pydantic import BaseModel


class Candidate(BaseModel):
    ligand_id: str
    smiles: str
    affinity_norm: float


class SafetyRequest(BaseModel):
    candidates: List[Candidate] = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="List of up to 2000 candidate molecules"
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

    def forward(self, x):

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
# LOAD SCALER
# ================================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"DNN model not found: {MODEL_PATH}"
    )

if not SCALER_PATH.exists():
    raise RuntimeError(
        f"Scaler not found: {SCALER_PATH}"
    )

scaler = joblib.load(
    SCALER_PATH
)

SCALER_FEATURES = getattr(
    scaler,
    "n_features_in_",
    None
)

if SCALER_FEATURES != 8:

    raise RuntimeError(
        "Unexpected scaler configuration. "
        f"Expected 8 descriptor features, "
        f"found {SCALER_FEATURES}."
    )

# ================================================================
# LOAD DNN CHECKPOINT
# ================================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

if not isinstance(checkpoint, dict):

    raise RuntimeError(
        "Invalid DNN checkpoint format."
    )

if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain "
        "'model_state_dict'."
    )

saved_input_dim = checkpoint.get(
    "input_dim"
)

saved_morgan_bits = checkpoint.get(
    "morgan_bits"
)

saved_endpoints = checkpoint.get(
    "endpoints"
)

if saved_input_dim != INPUT_DIM:

    raise RuntimeError(
        f"DNN input mismatch. "
        f"Checkpoint expects {saved_input_dim}, "
        f"API expects {INPUT_DIM}."
    )

if saved_morgan_bits != MORGAN_BITS:

    raise RuntimeError(
        f"Morgan fingerprint mismatch. "
        f"Checkpoint uses {saved_morgan_bits}, "
        f"API uses {MORGAN_BITS}."
    )

if saved_endpoints != ENDPOINTS:

    raise RuntimeError(
        "Endpoint configuration mismatch.\n"
        f"Checkpoint: {saved_endpoints}\n"
        f"API: {ENDPOINTS}"
    )

model = MultiTaskDNN(
    input_dim=INPUT_DIM,
    endpoints=ENDPOINTS,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.to(DEVICE)
model.eval()

# ================================================================
# ADMET-AI
# ================================================================

try:

    from admet_ai import ADMETModel

    admet_model = ADMETModel()

except Exception as e:

    raise RuntimeError(
        f"Could not load ADMET-AI: {e}"
    )

# ================================================================
# MORGAN FINGERPRINT
# ================================================================


def make_morgan_fingerprint(
    smiles: str
) -> np.ndarray:

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    generator = AllChem.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )

    fp = generator.GetFingerprint(
        mol
    )

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
# RDKit DESCRIPTORS
# ================================================================


def calculate_descriptors(
    smiles: str
) -> np.ndarray:

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    values = [

        Descriptors.MolWt(
            mol
        ),

        Crippen.MolLogP(
            mol
        ),

        Lipinski.NumHDonors(
            mol
        ),

        Lipinski.NumHAcceptors(
            mol
        ),

        Lipinski.NumRotatableBonds(
            mol
        ),

        rdMolDescriptors.CalcTPSA(
            mol
        ),

        Crippen.MolMR(
            mol
        ),

        rdMolDescriptors.CalcFractionCSP3(
            mol
        ),
    ]

    return np.asarray(
        values,
        dtype=np.float32,
    )


# ================================================================
# BUILD FEATURES
# ================================================================


def build_features(
    smiles_list: list[str],
) -> np.ndarray:

    fingerprints = []
    descriptors = []

    for smiles in smiles_list:

        fingerprints.append(
            make_morgan_fingerprint(
                smiles
            )
        )

        descriptors.append(
            calculate_descriptors(
                smiles
            )
        )

    X_fp = np.asarray(
        fingerprints,
        dtype=np.float32,
    )

    X_desc = np.asarray(
        descriptors,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # Scale ONLY descriptor columns.
    # Morgan fingerprints remain unchanged.
    # ------------------------------------------------------------

    X_desc_scaled = scaler.transform(
        X_desc
    ).astype(
        np.float32
    )

    X = np.concatenate(
        [
            X_fp,
            X_desc_scaled,
        ],
        axis=1,
    )

    if X.shape[1] != INPUT_DIM:

        raise RuntimeError(
            f"Feature dimension mismatch. "
            f"Generated {X.shape[1]}, "
            f"expected {INPUT_DIM}."
        )

    return X


# ================================================================
# DNN PREDICTION
# ================================================================


def dnn_predict(
    smiles_list: list[str]
) -> dict[str, np.ndarray]:

    X = build_features(
        smiles_list
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        outputs = model(
            X_tensor
        )

    predictions = {}

    for endpoint in ENDPOINTS:

        values = (
            outputs[endpoint]
            .detach()
            .cpu()
            .numpy()
        )

        if endpoint in CLASSIFICATION_ENDPOINTS:

            values = 1.0 / (
                1.0 +
                np.exp(-values)
            )

        predictions[endpoint] = values

    return predictions


# ================================================================
# ADMET-AI PREDICTION
# ================================================================


def admet_predict(
    smiles_list: list[str]
) -> pd.DataFrame:

    result = admet_model.predict(
        smiles_list
    )

    if not isinstance(
        result,
        pd.DataFrame
    ):

        result = pd.DataFrame(
            result
        )

    return result


# ================================================================
# SAFE FLOAT
# ================================================================


def safe_float(
    value,
    default=0.5
):

    try:

        if pd.isna(value):

            return default

        return float(value)

    except Exception:

        return default


# ================================================================
# GET ADMET COLUMN
# ================================================================


def get_admet_value(
    df: pd.DataFrame,
    row: int,
    columns: list[str],
    default=0.5,
):

    for column in columns:

        if column in df.columns:

            return safe_float(
                df.iloc[row][column],
                default,
            )

    return default


# ================================================================
# LIPINSKI
# ================================================================


def lipinski_violations(
    smiles: str
) -> int:

    mol = Chem.MolFromSmiles(
        smiles
    )

    if mol is None:

        return 4

    violations = 0

    if Descriptors.MolWt(mol) > 500:
        violations += 1

    if Crippen.MolLogP(mol) > 5:
        violations += 1

    if Lipinski.NumHDonors(mol) > 5:
        violations += 1

    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1

    return violations


# ================================================================
# SAFETY CALCULATION
# ================================================================


def calculate_safety(
    smiles_list: list[str],
    affinity_values: list[float],
):

    # ------------------------------------------------------------
    # DNN
    # ------------------------------------------------------------

    dnn = dnn_predict(
        smiles_list
    )

    # ------------------------------------------------------------
    # ADMET-AI
    # ------------------------------------------------------------

    admet = admet_predict(
        smiles_list
    )

    results = []

    for i, smiles in enumerate(
        smiles_list
    ):

        # ========================================================
        # DNN
        # ========================================================

        dnn_ames = float(
            dnn["AMES"][i]
        )

        dnn_herg = float(
            dnn["hERG"][i]
        )

        dnn_dili = float(
            dnn["DILI"][i]
        )

        dnn_clintox = float(
            dnn["ClinTox"][i]
        )

        dnn_carc = float(
            dnn["Carcinogens"][i]
        )

        dnn_bbb = float(
            dnn["BBB"][i]
        )

        dnn_hia = float(
            dnn["HIA"][i]
        )

        dnn_sol = float(
            dnn["Solubility"][i]
        )

        # ========================================================
        # ADMET-AI
        # ========================================================

        admet_ames = get_admet_value(
            admet,
            i,
            ["AMES"],
        )

        admet_herg = get_admet_value(
            admet,
            i,
            ["hERG"],
        )

        admet_dili = get_admet_value(
            admet,
            i,
            ["DILI"],
        )

        admet_clintox = get_admet_value(
            admet,
            i,
            ["ClinTox"],
        )

        admet_carc = get_admet_value(
            admet,
            i,
            [
                "Carcinogens_Lagunin",
                "Carcinogens",
            ],
        )

        admet_bbb = get_admet_value(
            admet,
            i,
            [
                "BBB_Martins",
                "BBB",
            ],
        )

        admet_hia = get_admet_value(
            admet,
            i,
            [
                "HIA_Hou",
                "HIA",
            ],
        )

        admet_sol = get_admet_value(
            admet,
            i,
            [
                "Solubility_AqSolDB",
                "Solubility",
            ],
        )

        # ========================================================
        # HYBRID DNN + ADMET-AI
        # ========================================================

        ames = (
            dnn_ames +
            admet_ames
        ) / 2.0

        herg = (
            dnn_herg +
            admet_herg
        ) / 2.0

        dili = (
            dnn_dili +
            admet_dili
        ) / 2.0

        clintox = (
            dnn_clintox +
            admet_clintox
        ) / 2.0

        carcinogens = (
            dnn_carc +
            admet_carc
        ) / 2.0

        bbb = (
            dnn_bbb +
            admet_bbb
        ) / 2.0

        hia = (
            dnn_hia +
            admet_hia
        ) / 2.0

        solubility = (
            dnn_sol +
            admet_sol
        ) / 2.0

        # ========================================================
        # TOXICITY
        # ========================================================

        toxicity_risk = float(
            np.mean(
                [
                    ames,
                    herg,
                    dili,
                    clintox,
                    carcinogens,
                ]
            )
        )

        toxicity_safety = (
            1.0 -
            toxicity_risk
        )

        # ========================================================
        # ADME
        # ========================================================

        adme_score = hia

        # ========================================================
        # QED
        # ========================================================

        qed = get_admet_value(
            admet,
            i,
            ["QED"],
        )

        # ========================================================
        # LIPINSKI
        # ========================================================

        violations = lipinski_violations(
            smiles
        )

        lipinski_score = (
            1.0 -
            violations / 4.0
        )

        drug_likeness = (
            0.667 * qed +
            0.333 * lipinski_score
        )

        # ========================================================
        # STRUCTURAL ALERTS
        # ========================================================

        pains = get_admet_value(
            admet,
            i,
            ["PAINS_alert"],
            default=0,
        )

        brenk = get_admet_value(
            admet,
            i,
            ["BRENK_alert"],
            default=0,
        )

        alert_count = (
            int(pains > 0) +
            int(brenk > 0)
        )

        alert_score = (
            1.0 -
            alert_count / 2.0
        )

        # ========================================================
        # HARD FAIL
        # ========================================================

        hard_fail = (
            ames >= 0.50 or
            herg >= 0.50
        )

        # ========================================================
        # FINAL SAFETY SCORE
        # ========================================================

        safety_score = (

            0.20 *
            drug_likeness

            +

            0.25 *
            adme_score

            +

            0.35 *
            toxicity_safety

            +

            0.20 *
            alert_score
        )

        # ========================================================
        # STATUS
        # ========================================================

        if hard_fail:

            status = "REJECT"

        elif safety_score >= 0.50:

            status = "PASS"

        else:

            status = "LOW_SAFETY"

        # ========================================================
        # REWARD
        # ========================================================

        affinity = float(
            affinity_values[i]
        )

        if hard_fail:

            reward = 0.0

        else:

            reward = (
                0.50 * affinity +
                0.50 * safety_score
            )

        # ========================================================
        # RESULT
        # ========================================================

        results.append(
            {
                "smiles": smiles,

                "affinity_norm": round(
                    affinity,
                    6,
                ),

                "AMES": round(
                    ames,
                    6,
                ),

                "hERG": round(
                    herg,
                    6,
                ),

                "DILI": round(
                    dili,
                    6,
                ),

                "ClinTox": round(
                    clintox,
                    6,
                ),

                "Carcinogens": round(
                    carcinogens,
                    6,
                ),

                "BBB": round(
                    bbb,
                    6,
                ),

                "HIA": round(
                    hia,
                    6,
                ),

                "Solubility": round(
                    solubility,
                    6,
                ),

                "toxicity_risk": round(
                    toxicity_risk,
                    6,
                ),

                "toxicity_safety": round(
                    toxicity_safety,
                    6,
                ),

                "adme_score": round(
                    adme_score,
                    6,
                ),

                "QED": round(
                    qed,
                    6,
                ),

                "lipinski_violations": (
                    violations
                ),

                "drug_likeness": round(
                    drug_likeness,
                    6,
                ),

                "PAINS_alert": int(
                    pains > 0
                ),

                "BRENK_alert": int(
                    brenk > 0
                ),

                "alert_score": round(
                    alert_score,
                    6,
                ),

                "hard_fail": bool(
                    hard_fail
                ),

                "safety_score": round(
                    safety_score,
                    6,
                ),

                "status": status,

                "reward": round(
                    reward,
                    6,
                ),
            }
        )

    return results


# ================================================================
# HEALTH
# ================================================================


@app.get("/health")
def health():

    return {
        "status": "healthy",

        "module": (
            "Module 3 - Deep Safety Model"
        ),

        "model": (
            "MultiTaskDNN + ADMET-AI"
        ),

        "device": str(DEVICE),

        "input_features": INPUT_DIM,

        "morgan_bits": MORGAN_BITS,

        "rdkit_descriptors": len(
            DESCRIPTOR_NAMES
        ),

        "scaled_features": (
            SCALER_FEATURES
        ),

        "endpoints": ENDPOINTS,
    }


# ================================================================
# SAFETY
# ================================================================


@app.post("/safety")
def safety(
    request: SafetyRequest
):

    try:

        ligand_ids = [
            candidate.ligand_id
            for candidate in request.candidates
        ]

        smiles_list = [
            candidate.smiles
            for candidate in request.candidates
        ]

        affinity_values = [
            candidate.affinity_norm
            for candidate in request.candidates
        ]

        # --------------------------------------------------------
        # Duplicate IDs
        # --------------------------------------------------------

        if len(ligand_ids) != len(
            set(ligand_ids)
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Duplicate ligand_id detected."
                ),
            )

        # --------------------------------------------------------
        # Validate SMILES
        # --------------------------------------------------------

        for smiles in smiles_list:

            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is None:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid SMILES: {smiles}"
                    ),
                )

        # --------------------------------------------------------
        # Predict
        # --------------------------------------------------------

        results = calculate_safety(
            smiles_list,
            affinity_values,
        )

        # --------------------------------------------------------
        # Add ligand IDs
        # --------------------------------------------------------

        for i in range(
            len(results)
        ):

            results[i]["ligand_id"] = (
                ligand_ids[i]
            )

        # --------------------------------------------------------
        # Rank
        # --------------------------------------------------------

        results.sort(
            key=lambda x: (
                x["hard_fail"],
                -x["reward"],
            )
        )

        # --------------------------------------------------------
        # Ranking number
        # --------------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1,
        ):

            result["rank"] = rank

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        pass_count = sum(
            result["status"] == "PASS"
            for result in results
        )

        reject_count = sum(
            result["status"] == "REJECT"
            for result in results
        )

        low_safety_count = sum(
            result["status"] == "LOW_SAFETY"
            for result in results
        )

        return {

            "module": "Module 3",

            "status": "success",

            "total_candidates": len(
                results
            ),

            "pass": pass_count,

            "reject": reject_count,

            "low_safety": low_safety_count,

            "results": results,
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ================================================================
# ROOT
# ================================================================


@app.get("/")
def root():

    return {

        "module": (
            "Module 3 - Deep Safety Model"
        ),

        "status": "running",

        "endpoints": {
            "health": "/health",
            "safety": "/safety",
        },
    }


# ================================================================
# DIRECT EXECUTION
# ================================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )