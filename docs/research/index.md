# Research

What is known about decision models, Jev, Mimir and HRM-Text, with sources.

## Architecture

* [HRM-Text](/research/hrm-text.md) - Hierarchical recurrent language model with slow H-module and fast L-module cycles; basis of Mimir.

## Concept

* [Decision Models (System One models)](/research/decision-models.md) - Non-generative models that read text and return typed, calibrated probabilistic decisions instead of tokens.

## Model

* [DFM Mimir v1.5](/research/mimir-v1-5.md) - Final epoch-10 EMA checkpoint of the XL HRM-Text lineage; the chosen starting point for Mimir-Decide.
* [DFM Mimir v1](/research/mimir-v1.md) - Open 1B-parameter HRM-Text model from Danish Foundation Models, trained on permissible data; candidate base for a decision model.

## Product

* [Jev (TypeSafe AI)](/research/jev.md) - First commercial decision model; API returning calibrated Choice/Score/Noul outputs at 70-500 ms.
* [Laya](/research/laya.md) - Small Jev-family decision model, self-hostable on a T4 GPU, with explicit Brier-score training.

## Reference

* [RLCD and calibration training](/research/calibration-training.md) - Training objectives for calibrated decisions: RLCD (Jev), RLCR, Brier-score rewards, temperature scaling, RLHV.
* [Colleague recommendations (Mimir training lead)](/research/colleague-recommendations.md) - What the colleague who has been training Mimir recommended for turning it into a decision model, and which parts we adopted.
* [Open questions](/research/open-questions.md) - Unknowns about Jev and about adapting Mimir; things to verify.
* [Sources](/research/sources.md) - Sources consulted on 2026-10-03 and the depth of verification for each.
