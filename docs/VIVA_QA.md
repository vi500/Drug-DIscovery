# Viva Q&A

Q: Why multi-task learning?
A: A shared molecular representation lets related ADME/Tox tasks learn common chemistry while retaining endpoint-specific heads.

Q: Why Morgan fingerprints?
A: They provide a reproducible fixed-length representation of local molecular substructures.

Q: Why scaffold split?
A: Random splits can place close analogues in train and test. Scaffold splitting gives a stricter chemical generalisation test.

Q: Why use ADMET-AI?
A: It is a real pretrained baseline trained on TDC-derived ADMET tasks. We do not claim it is ground truth.

Q: What is the safety score?
A: A transparent prioritisation score combining drug-likeness, ADME, toxicity and structural alerts.

Q: Is this clinical safety?
A: No. It is computational prioritisation only and must be experimentally validated.
