📄 Prescription NLP Extraction Pipeline

🚀 Overview

This project focuses on extracting structured information from noisy medical prescription text using a hybrid NLP approach (rule-based + machine learning).

The pipeline processes raw prescription data and converts it into structured JSON with fields such as:

Medicine Name
Dosage
Frequency
Duration
Strength
Form
Notes

🧠 Approach

A hybrid pipeline is used to balance accuracy, interpretability, and efficiency:

1. Preprocessing
Normalize text (lowercase, spacing fixes)
Expand abbreviations:
OD → once daily
BD → twice daily
Tab → tablet
Handle noisy inputs:
Typos (e.g., Paracitamol → Paracetamol)
Fused tokens (e.g., Ome40mg)
2. Rule-Based Extraction (Regex + Lookup)

Used for deterministic fields:

Strength → \d+ mg
Duration → \d+ days / weeks
Dosage patterns → 1-0-1, 1-1-1
Frequency mapping:
OD, BD, TDS, HS, etc.
Form detection:
Tablet, Capsule, Syrup, Drops
3. ML-Based Extraction
TF-IDF + Logistic Regression model
Used for medicine name extraction
Handles:
Abbreviations (PCM, DOXY, Panto)
Variants & typos
4. Post-processing
Conflict resolution (e.g., OD vs bedtime)
Confidence handling
Structured JSON formatting
