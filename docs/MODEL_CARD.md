# Module 3 Model Card

## Purpose
Computational prioritisation of small molecules using drug-likeness, pretrained ADMET-AI, and a custom multi-task DNN.

## Main model
Shared PyTorch encoder over RDKit Morgan fingerprints plus eight RDKit descriptors, with one head per endpoint.

## Endpoints
AMES, hERG, DILI, ClinTox, Carcinogens, BBB, HIA, Solubility.

## Evaluation
Classification: AUROC and AUPRC. Regression: RMSE and R2. Scaffold split is used.

## Limitations
Predictions are computational assay-level estimates. They are not clinical safety determinations and require experimental validation. Dataset shift and applicability-domain limitations can materially reduce accuracy.

## Reproducibility
Seed, configuration, model checkpoint, scaler, metrics, package versions, and run outputs are archived.
