from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from rdkit import Chem


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPARISON_FILE = PROJECT_ROOT / "outputs" / "safety_model_comparison.csv"
DRUG_LIKENESS_FILE = PROJECT_ROOT / "outputs" / "drug_likeness.csv"
ADMET_AI_FILE = PROJECT_ROOT / "outputs" / "admet_ai_predictions.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "safety_scores.csv"


# ============================================================
# ENDPOINTS
# ============================================================

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

TOX_ENDPOINTS = [
    "AMES",
    "hERG",
    "DILI",
    "ClinTox",
    "Carcinogens",
]


# ============================================================
# CONFIGURATION
# ============================================================

AMES_HARD_FAIL_THRESHOLD = 0.50
HERG_HARD_FAIL_THRESHOLD = 0.50

WEIGHT_DRUG_LIKENESS = 0.20
WEIGHT_ADME = 0.25
WEIGHT_TOXICITY = 0.35
WEIGHT_ALERTS = 0.20


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)
    print(message)
    print("=" * 70)
    sys.exit(1)


def require_file(path, description):
    if not path.exists():
        fail(
            f"{description} not found:\n{path}"
        )


def require_columns(df, columns, name):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        fail(
            f"{name} is missing columns:\n"
            + "\n".join(
                f"  - {c}" for c in missing
            )
        )


def check_ids(df, name):
    if "candidate_id" not in df.columns:
        fail(
            f"{name} does not contain candidate_id."
        )

    if df["candidate_id"].isna().any():
        fail(
            f"{name} contains missing candidate_id."
        )

    if df["candidate_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["candidate_id"].duplicated(keep=False),
                ["candidate_id"]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

        fail(
            f"{name} contains duplicate candidate_id:\n"
            f"{duplicates}"
        )


def numeric_column(df, column):
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    if values.isna().any():
        bad = df.loc[
            values.isna(),
            ["candidate_id", column]
        ]

        fail(
            f"Invalid numeric values in {column}:\n"
            + bad.to_string(index=False)
        )

    return values.astype(float)


def probability_series(values, index):
    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    if values.isna().any():
        fail(
            "Missing numeric prediction encountered."
        )

    clipped = np.clip(
        values.to_numpy(dtype=float),
        0.0,
        1.0
    )

    return pd.Series(
        clipped,
        index=index,
        dtype=float
    )


# ============================================================
# NEW: CANONICAL SMILES
# ============================================================

def canonical_smiles(smiles):
    """
    Convert a SMILES string to canonical RDKit SMILES.

    This allows chemically equivalent SMILES representations
    to be compared correctly.

    Example:
        C1=CC=CC=C1
        c1ccccc1

    Both represent benzene and therefore produce the same
    canonical representation.
    """

    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True
    )


def check_smiles_match(
    df,
    first_column,
    second_column,
    description
):
    """
    Compare two SMILES columns using canonical RDKit SMILES.

    This is safer than comparing raw SMILES strings because
    the same molecule can have different valid SMILES forms.
    """

    first_canonical = (
        df[first_column]
        .astype(str)
        .map(canonical_smiles)
    )

    second_canonical = (
        df[second_column]
        .astype(str)
        .map(canonical_smiles)
    )

    invalid_first = first_canonical.isna()
    invalid_second = second_canonical.isna()

    if invalid_first.any():
        bad = df.loc[
            invalid_first,
            ["candidate_id", first_column]
        ]

        fail(
            f"Invalid SMILES in {description} "
            f"({first_column}):\n\n"
            + bad.to_string(index=False)
        )

    if invalid_second.any():
        bad = df.loc[
            invalid_second,
            ["candidate_id", second_column]
        ]

        fail(
            f"Invalid SMILES in {description} "
            f"({second_column}):\n\n"
            + bad.to_string(index=False)
        )

    mismatch = (
        first_canonical.to_numpy()
        !=
        second_canonical.to_numpy()
    )

    if mismatch.any():

        bad_columns = [
            "candidate_id",
            first_column,
            second_column,
        ]

        bad = df.loc[
            mismatch,
            bad_columns
        ].copy()

        bad["canonical_first"] = first_canonical.loc[
            mismatch
        ].to_numpy()

        bad["canonical_second"] = second_canonical.loc[
            mismatch
        ].to_numpy()

        fail(
            f"{description}.\n\n"
            f"The following candidates represent different "
            f"molecular structures:\n\n"
            + bad.to_string(index=False)
        )

    print(
        f"SMILES consistency check passed: "
        f"{len(df)} candidates."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Module 3 Step 6 - Final Safety Score"
    )

    parser.add_argument(
        "--cns-target",
        action="store_true",
        help="Use BBB together with HIA for ADME scoring."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("MODULE 3 - STEP 6")
    print("FINAL SAFETY SCORE + HARD FILTER")
    print("=" * 70)

    print("Seed:", args.seed)
    print("CNS target:", args.cns_target)

    # ========================================================
    # FILE CHECK
    # ========================================================

    require_file(
        COMPARISON_FILE,
        "Step 5 comparison output"
    )

    require_file(
        DRUG_LIKENESS_FILE,
        "Step 2 drug-likeness output"
    )

    require_file(
        ADMET_AI_FILE,
        "Step 3 ADMET-AI output"
    )

    # ========================================================
    # LOAD
    # ========================================================

    comparison = pd.read_csv(
        COMPARISON_FILE
    )

    drug = pd.read_csv(
        DRUG_LIKENESS_FILE
    )

    admet = pd.read_csv(
        ADMET_AI_FILE
    )

    print(
        "Comparison rows:",
        len(comparison)
    )

    print(
        "Drug-likeness rows:",
        len(drug)
    )

    print(
        "ADMET-AI rows:",
        len(admet)
    )

    # ========================================================
    # VALIDATE IDs
    # ========================================================

    check_ids(
        comparison,
        "Step 5 comparison"
    )

    check_ids(
        drug,
        "Drug-likeness"
    )

    check_ids(
        admet,
        "ADMET-AI"
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    comparison_columns = [
        "candidate_id",
        "smiles",
    ]

    for endpoint in ENDPOINTS:

        comparison_columns.extend(
            [
                f"{endpoint}_DNN",
                f"{endpoint}_ADMET_AI",
            ]
        )

    require_columns(
        comparison,
        comparison_columns,
        "Step 5 comparison"
    )

    require_columns(
        drug,
        [
            "candidate_id",
            "smiles",
            "QED",
            "LipinskiViolations",
        ],
        "Drug-likeness"
    )

    require_columns(
        admet,
        [
            "candidate_id",
            "smiles",
            "PAINS_alert",
            "BRENK_alert",
        ],
        "ADMET-AI"
    )

    # ========================================================
    # MERGE STEP 5 + DRUG LIKENESS
    # ========================================================

    print()
    print(
        "Merging Step 5 and drug-likeness by candidate_id..."
    )

    df = comparison.merge(
        drug[
            [
                "candidate_id",
                "smiles",
                "QED",
                "LipinskiViolations",
            ]
        ],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_comparison",
            "_drug"
        )
    )

    if len(df) != len(comparison):
        fail(
            "Candidate count changed during merge."
        )

    # ========================================================
    # FIXED SMILES CHECK
    # ========================================================

    check_smiles_match(
        df,
        "smiles_comparison",
        "smiles_drug",
        "SMILES mismatch between Step 2 and Step 5"
    )

    df["smiles"] = (
        df["smiles_comparison"]
        .astype(str)
    )

    print(
        "Merged rows:",
        len(df)
    )

    # ========================================================
    # FINAL ENDPOINT VALUES
    # ========================================================

    print()
    print(
        "Creating final endpoint predictions..."
    )

    for endpoint in ENDPOINTS:

        dnn_col = f"{endpoint}_DNN"
        admet_col = f"{endpoint}_ADMET_AI"

        dnn = numeric_column(
            df,
            dnn_col
        )

        admet_values = numeric_column(
            df,
            admet_col
        )

        df[f"Final_{endpoint}"] = (
            dnn + admet_values
        ) / 2.0

    # ========================================================
    # TOXICITY
    # ========================================================

    print(
        "Calculating toxicity safety..."
    )

    tox_series = []

    for endpoint in TOX_ENDPOINTS:

        values = probability_series(
            df[f"Final_{endpoint}"],
            df.index
        )

        tox_series.append(
            values
        )

    toxicity_matrix = np.vstack(
        [
            x.to_numpy()
            for x in tox_series
        ]
    )

    toxicity_risk = np.mean(
        toxicity_matrix,
        axis=0
    )

    toxicity_safety = pd.Series(
        1.0 - toxicity_risk,
        index=df.index
    )

    toxicity_safety = probability_series(
        toxicity_safety,
        df.index
    )

    # ========================================================
    # ADME
    # ========================================================

    print(
        "Calculating ADME score..."
    )

    hia = probability_series(
        df["Final_HIA"],
        df.index
    )

    bbb = probability_series(
        df["Final_BBB"],
        df.index
    )

    if args.cns_target:

        adme_score = (
            0.50 * hia
            +
            0.50 * bbb
        )

        print(
            "ADME: HIA + BBB"
        )

    else:

        adme_score = hia

        print(
            "ADME: HIA"
        )

    adme_score = probability_series(
        adme_score,
        df.index
    )

    # ========================================================
    # DRUG LIKENESS
    # ========================================================

    print(
        "Calculating drug-likeness..."
    )

    qed = probability_series(
        numeric_column(
            df,
            "QED"
        ),
        df.index
    )

    lipinski = numeric_column(
        df,
        "LipinskiViolations"
    )

    lipinski_score = probability_series(
        1.0 - (lipinski / 4.0),
        df.index
    )

    drug_likeness_score = (
        0.667 * qed
        +
        0.333 * lipinski_score
    )

    drug_likeness_score = probability_series(
        drug_likeness_score,
        df.index
    )

    # ========================================================
    # STRUCTURAL ALERTS
    # ========================================================

    print(
        "Calculating structural alerts..."
    )

    alerts = admet[
        [
            "candidate_id",
            "smiles",
            "PAINS_alert",
            "BRENK_alert",
        ]
    ].copy()

    df = df.merge(
        alerts,
        on="candidate_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "",
            "_alert"
        )
    )

    if len(df) != len(comparison):
        fail(
            "Candidate count changed during alert merge."
        )

    # ========================================================
    # FIXED SECOND SMILES CHECK
    # ========================================================

    check_smiles_match(
        df,
        "smiles",
        "smiles_alert",
        "SMILES mismatch during alert merge"
    )

    pains = numeric_column(
        df,
        "PAINS_alert"
    )

    brenk = numeric_column(
        df,
        "BRENK_alert"
    )

    alert_count = (
        (pains > 0).astype(int)
        +
        (brenk > 0).astype(int)
    )

    alert_score = probability_series(
        1.0 - (alert_count / 2.0),
        df.index
    )

    # ========================================================
    # FINAL SAFETY SCORE
    # ========================================================

    print()
    print(
        "Calculating final safety score..."
    )

    safety_score = (
        WEIGHT_DRUG_LIKENESS
        * drug_likeness_score
        +
        WEIGHT_ADME
        * adme_score
        +
        WEIGHT_TOXICITY
        * toxicity_safety
        +
        WEIGHT_ALERTS
        * alert_score
    )

    safety_score = probability_series(
        safety_score,
        df.index
    )

    # ========================================================
    # HARD FILTER
    # ========================================================

    print(
        "Applying hard safety filters..."
    )

    ames_risk = probability_series(
        df["Final_AMES"],
        df.index
    )

    herg_risk = probability_series(
        df["Final_hERG"],
        df.index
    )

    hard_fail = (
        (ames_risk >= AMES_HARD_FAIL_THRESHOLD)
        |
        (herg_risk >= HERG_HARD_FAIL_THRESHOLD)
    )

    hard_fail = pd.Series(
        hard_fail,
        index=df.index,
        dtype=bool
    )

    # ========================================================
    # HARD FAIL REASONS
    # ========================================================

    reasons = []

    for i in df.index:

        candidate_reasons = []

        if (
            ames_risk.loc[i]
            >=
            AMES_HARD_FAIL_THRESHOLD
        ):

            candidate_reasons.append(
                "AMES>=0.50"
            )

        if (
            herg_risk.loc[i]
            >=
            HERG_HARD_FAIL_THRESHOLD
        ):

            candidate_reasons.append(
                "hERG>=0.50"
            )

        if candidate_reasons:

            reasons.append(
                ";".join(candidate_reasons)
            )

        else:

            reasons.append("")

    # ========================================================
    # STATUS
    # ========================================================

    status = np.where(
        hard_fail.to_numpy(),
        "REJECT",
        np.where(
            safety_score.to_numpy() >= 0.50,
            "PASS",
            "LOW_SAFETY"
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    output = pd.DataFrame(
        {
            "candidate_id": (
                df["candidate_id"]
                .astype(str)
            ),

            "smiles": (
                df["smiles"]
                .astype(str)
            ),

            "safety_score": (
                safety_score
                .to_numpy()
            ),

            "safety_status": status,

            "hard_fail": (
                hard_fail
                .to_numpy()
            ),

            "hard_fail_reason": reasons,
        }
    )

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    expected_columns = [
        "candidate_id",
        "smiles",
        "safety_score",
        "safety_status",
        "hard_fail",
        "hard_fail_reason",
    ]

    if list(output.columns) != expected_columns:

        fail(
            "Incorrect final output columns."
        )

    if len(output) != len(comparison):

        fail(
            "Final candidate count changed."
        )

    if output["candidate_id"].duplicated().any():

        fail(
            "Duplicate candidate IDs in final output."
        )

    if output["safety_score"].isna().any():

        fail(
            "Missing safety scores."
        )

    if (
        (output["safety_score"] < 0).any()
        or
        (output["safety_score"] > 1).any()
    ):

        fail(
            "Safety score outside [0,1]."
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 6 COMPLETE")
    print("=" * 70)

    print(
        "Candidates:",
        len(output)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print()
    print(
        "Safety score range:",
        f"{output['safety_score'].min():.4f}",
        "to",
        f"{output['safety_score'].max():.4f}"
    )

    print()
    print(
        "Mean safety score:",
        f"{output['safety_score'].mean():.4f}"
    )

    print()
    print("Status counts:")

    print(
        output[
            "safety_status"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Hard failures:",
        int(
            output["hard_fail"].sum()
        )
    )

    print()
    print("Safety scores:")

    # For very large runs such as 10K, avoid dumping
    # thousands of rows unnecessarily.
    if len(output) <= 100:

        print(
            output.to_string(
                index=False
            )
        )

    else:

        print(
            output.head(20).to_string(
                index=False
            )
        )

        print()
        print(
            f"... {len(output) - 20} additional "
            "candidates saved to CSV."
        )

    print()
    print("=" * 70)
    print("STEP 6 FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()