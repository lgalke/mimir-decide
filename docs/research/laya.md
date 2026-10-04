---
type: Product
title: Laya
description: Small Jev-family decision model, self-hostable on a T4 GPU, with explicit Brier-score training.
tags: [decision-model, laya, calibration]
timestamp: 2026-10-03T00:00:00Z
---

# Laya

Described in the pentest-harness paper (arXiv 2609.28940) alongside [Jev](/research/jev.md).

- Latency 33-40 ms on a T4 GPU.
- Training documented as optimising a proper scoring rule (Brier score), "calibrated by construction".
- Raw ECE 0.466, **0.081 after temperature scaling**. This shows post-hoc calibration matters a lot.
- Caveat from the paper: Jev and Laya metrics are on different datasets and not comparable; no transfer of calibration to the pentest domain was verified.

Laya is the closest public reference point for what a self-hosted decision model of the size we would build looks like. Details beyond the above were not verified.
