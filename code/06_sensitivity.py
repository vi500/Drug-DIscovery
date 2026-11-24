from pathlib import Path
import argparse
import pandas as pd
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code import config

def calculate_safety_score(df, w_qed, w_lipinski, w_adme, w_tox, w_alert):
    scores = []

    for _, row in df.iterrows():

        qed = float(row.get("QED", 0.0))
        lipinski_violations = float(row.get("LipinskiViolations", 0.0))

        lipinski_score = max(
            0.0,
            1.0 - min(lipinski_violations / 4.0, 1.0)
        )

        # ADME component
        adme_values = []

        for col in ["HIA", "BBB"]:
            if col in df.columns:
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(value):
                    adme_values.append(float(value))

        if adme_values:
            adme_score = float(np.mean(adme_values))
        else:
            adme_score = 0.5

        # Toxicity component
        tox_values = []

        for col in ["AMES", "hERG", "DILI", "ClinTox", "Carcinogens"]:
            if col in df.columns:
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(value):
                    tox_values.append(1.0 - float(value))

        if tox_values:
            tox_score = float(np.mean(tox_values))
        else:
            tox_score = 0.5

        # Alert component
        alert_values = []

        for col in ["PAINS_alert", "BRENK_alert", "NIH_alert"]:
            if col in df.columns:
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(value):
                    alert_values.append(1.0 - float(value))

        if alert_values:
            alert_score = float(np.mean(alert_values))
        else:
            alert_score = 1.0

        score = (
            w_qed * qed
            + w_lipinski * lipinski_score
            + w_adme * adme_score
            + w_tox * tox_score
            + w_alert * alert_score
        )

        scores.append(float(np.clip(score, 0.0, 1.0)))

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Module 3 - Safety Score Sensitivity Analysis"
    )

    parser.add_argument(
        "--input",
        default=str(config.SAFETY_COMPARISON_FILE),
        help="Input safety/model comparison CSV"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Random seed"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    input_file = Path(args.input)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    required = {"candidate_id", "smiles"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Use DNN/ADMET comparison columns to create stable endpoint values.
    endpoint_map = {
        "AMES": "AMES_DNN",
        "hERG": "hERG_DNN",
        "DILI": "DILI_DNN",
        "ClinTox": "ClinTox_DNN",
        "Carcinogens": "Carcinogens_DNN",
        "BBB": "BBB_DNN",
        "HIA": "HIA_DNN",
        "Solubility": "Solubility_DNN",
    }

    for endpoint, column in endpoint_map.items():
        if column in df.columns:
            df[endpoint] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Drug-likeness values are not in Step 5, so load them if available.
    drug_file = config.DRUG_LIKENESS_FILE

    if drug_file.exists():
        drug = pd.read_csv(drug_file)

        keep = [
            c for c in [
                "candidate_id",
                "QED",
                "LipinskiViolations"
            ]
            if c in drug.columns
        ]

        if "candidate_id" in keep:
            df = df.merge(
                drug[keep],
                on="candidate_id",
                how="left",
                validate="one_to_one"
            )

    if "QED" not in df.columns:
        df["QED"] = 0.5

    if "LipinskiViolations" not in df.columns:
        df["LipinskiViolations"] = 0.0

    # Add alert columns if available from ADMET-AI.
    admet_file = config.ADMET_AI_FILE

    if admet_file.exists():
        admet = pd.read_csv(admet_file)

        alert_cols = [
            "candidate_id",
            "PAINS_alert",
            "BRENK_alert",
            "NIH_alert",
        ]

        available = [
            c for c in alert_cols
            if c in admet.columns
        ]

        if "candidate_id" in available:
            df = df.merge(
                admet[available],
                on="candidate_id",
                how="left",
                validate="one_to_one",
                suffixes=("", "_admet")
            )

    # Baseline configuration
    baseline = np.array([
        config.W_QED,
        config.W_LIPINSKI,
        config.W_ADME,
        config.W_TOX,
        config.W_ALERT,
    ])

    # Sensitivity scenarios.
    scenarios = [
        ("baseline", *baseline),

        ("toxicity_high",
         0.15, 0.10, 0.20, 0.45, 0.10),

        ("adme_high",
         0.15, 0.10, 0.40, 0.25, 0.10),

        ("qed_high",
         0.35, 0.10, 0.20, 0.25, 0.10),

        ("alerts_high",
         0.15, 0.10, 0.20, 0.30, 0.25),

        ("balanced",
         0.20, 0.20, 0.20, 0.20, 0.20),
    ]

    results = []

    for name, w_qed, w_lipinski, w_adme, w_tox, w_alert in scenarios:

        scores = calculate_safety_score(
            df,
            w_qed,
            w_lipinski,
            w_adme,
            w_tox,
            w_alert
        )

        temp = df[["candidate_id", "smiles"]].copy()
        temp["scenario"] = name

        temp["safety_score"] = scores

        temp["rank"] = (
            temp["safety_score"]
            .rank(
                ascending=False,
                method="min"
            )
            .astype(int)
        )

        results.append(temp)

    output = pd.concat(
        results,
        ignore_index=True
    )

    config.OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        config.SENSITIVITY_FILE,
        index=False
    )

    print("=" * 70)
    print("MODULE 3 - STEP 6 SENSITIVITY ANALYSIS")
    print("=" * 70)

    print(f"Candidates: {df['candidate_id'].nunique()}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Seed: {args.seed}")

    print(
        f"Output: {config.SENSITIVITY_FILE}"
    )

    print("\nMean safety score by scenario:")

    summary = (
        output
        .groupby("scenario")["safety_score"]
        .mean()
    )

    for scenario, value in summary.items():
        print(
            f"{scenario:20s}: {value:.4f}"
        )

    print("\nSTEP 6 SENSITIVITY COMPLETE")


if __name__ == "__main__":
    main()