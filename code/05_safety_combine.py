from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DNN_FILE = PROJECT_ROOT / "outputs" / "dnn_candidate_predictions.csv"
ADMET_FILE = PROJECT_ROOT / "outputs" / "admet_ai_predictions.csv"

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "safety_model_comparison.csv"


# DNN endpoint -> actual ADMET-AI endpoint
ENDPOINT_MAP = {
    "AMES": "AMES",
    "hERG": "hERG",
    "DILI": "DILI",
    "ClinTox": "ClinTox",
    "Carcinogens": "Carcinogens_Lagunin",
    "BBB": "BBB_Martins",
    "HIA": "HIA_Hou",
    "Solubility": "Solubility_AqSolDB",
}


def fail(message):
    print()
    print("ERROR:", message)
    sys.exit(1)


def main():

    print("=" * 70)
    print("MODULE 3 - STEP 5")
    print("DNN + ADMET-AI COMPARISON")
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not DNN_FILE.exists():
        fail(f"DNN prediction file not found:\n{DNN_FILE}")

    if not ADMET_FILE.exists():
        fail(f"ADMET-AI prediction file not found:\n{ADMET_FILE}")

    # ========================================================
    # LOAD
    # ========================================================

    dnn = pd.read_csv(DNN_FILE)
    admet = pd.read_csv(ADMET_FILE)

    print("DNN rows:", len(dnn))
    print("ADMET-AI rows:", len(admet))

    # ========================================================
    # CHECK ID COLUMNS
    # ========================================================

    for column in ["candidate_id", "smiles"]:

        if column not in dnn.columns:
            fail(f"DNN file missing column: {column}")

        if column not in admet.columns:
            fail(f"ADMET-AI file missing column: {column}")

    # ========================================================
    # CHECK ENDPOINT COLUMNS
    # ========================================================

    for dnn_endpoint, admet_endpoint in ENDPOINT_MAP.items():

        if dnn_endpoint not in dnn.columns:
            fail(
                f"DNN endpoint missing: {dnn_endpoint}"
            )

        if admet_endpoint not in admet.columns:
            fail(
                f"ADMET-AI endpoint missing: {admet_endpoint}"
            )

    # ========================================================
    # CHECK DUPLICATES
    # ========================================================

    if dnn["candidate_id"].duplicated().any():
        fail(
            "Duplicate candidate_id found in DNN predictions."
        )

    if admet["candidate_id"].duplicated().any():
        fail(
            "Duplicate candidate_id found in ADMET-AI predictions."
        )

    # ========================================================
    # CHECK CANDIDATE IDS
    # ========================================================

    dnn_ids = set(
        dnn["candidate_id"].astype(str)
    )

    admet_ids = set(
        admet["candidate_id"].astype(str)
    )

    if dnn_ids != admet_ids:

        missing_in_admet = sorted(
            dnn_ids - admet_ids
        )

        missing_in_dnn = sorted(
            admet_ids - dnn_ids
        )

        fail(
            "Candidate ID sets do not match.\n"
            f"Missing in ADMET-AI: {missing_in_admet}\n"
            f"Missing in DNN: {missing_in_dnn}"
        )

    # ========================================================
    # PREPARE DNN TABLE
    # ========================================================

    dnn_part = dnn[
        ["candidate_id", "smiles"]
        + list(ENDPOINT_MAP.keys())
    ].copy()

    # Rename DNN prediction columns explicitly.
    dnn_rename = {
        endpoint: f"{endpoint}_DNN"
        for endpoint in ENDPOINT_MAP.keys()
    }

    dnn_part = dnn_part.rename(
        columns=dnn_rename
    )

    # ========================================================
    # PREPARE ADMET-AI TABLE
    # ========================================================

    admet_columns = [
        "candidate_id",
        "smiles"
    ] + list(
        ENDPOINT_MAP.values()
    )

    admet_part = admet[
        admet_columns
    ].copy()

    # Rename ADMET-AI columns explicitly
    admet_rename = {
        admet_endpoint: f"{dnn_endpoint}_ADMET_AI"
        for dnn_endpoint, admet_endpoint
        in ENDPOINT_MAP.items()
    }

    admet_part = admet_part.rename(
        columns=admet_rename
    )

    # ========================================================
    # MERGE BY candidate_id
    # ========================================================

    merged = dnn_part.merge(
        admet_part,
        on="candidate_id",
        how="inner",
        validate="one_to_one",
        suffixes=(
            "_dnn",
            "_admet"
        )
    )

    print("Merged rows:", len(merged))

    if len(merged) != len(dnn):

        fail(
            "Merge changed candidate count.\n"
            f"DNN rows: {len(dnn)}\n"
            f"Merged rows: {len(merged)}"
        )

    # ========================================================
    # VERIFY SMILES
    # ========================================================

    if not (
        merged["smiles_dnn"].astype(str)
        ==
        merged["smiles_admet"].astype(str)
    ).all():

        bad = merged[
            merged["smiles_dnn"].astype(str)
            !=
            merged["smiles_admet"].astype(str)
        ][
            [
                "candidate_id",
                "smiles_dnn",
                "smiles_admet"
            ]
        ]

        print()
        print(
            bad.to_string(index=False)
        )

        fail(
            "SMILES mismatch between DNN and ADMET-AI."
        )

    # ========================================================
    # CREATE RESULT
    # ========================================================

    result = pd.DataFrame()

    result["candidate_id"] = (
        merged["candidate_id"]
    )

    result["smiles"] = (
        merged["smiles_dnn"]
    )

    # ========================================================
    # COMPARE ALL ENDPOINTS
    # ========================================================

    for endpoint in ENDPOINT_MAP.keys():

        dnn_column = f"{endpoint}_DNN"
        admet_column = f"{endpoint}_ADMET_AI"

        if dnn_column not in merged.columns:

            fail(
                f"DNN comparison column missing: "
                f"{dnn_column}"
            )

        if admet_column not in merged.columns:

            fail(
                f"ADMET-AI comparison column missing: "
                f"{admet_column}"
            )

        dnn_values = pd.to_numeric(
            merged[dnn_column],
            errors="raise"
        )

        admet_values = pd.to_numeric(
            merged[admet_column],
            errors="raise"
        )

        result[dnn_column] = (
            dnn_values.values
        )

        result[admet_column] = (
            admet_values.values
        )

        result[
            f"{endpoint}_ABS_DIFF"
        ] = (
            dnn_values
            .sub(admet_values)
            .abs()
            .values
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)

    print(
        "Candidates compared:",
        len(result)
    )

    print(
        "Endpoints compared:",
        len(ENDPOINT_MAP)
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print()
    print("Mean absolute difference:")

    for endpoint in ENDPOINT_MAP.keys():

        diff_column = (
            f"{endpoint}_ABS_DIFF"
        )

        print(
            f"{endpoint:15s}: "
            f"{result[diff_column].mean():.6f}"
        )

    print()
    print("Comparison preview:")
    print()

    print(
        result.to_string(
            index=False
        )
    )

    print("=" * 70)


if __name__ == "__main__":
    main()