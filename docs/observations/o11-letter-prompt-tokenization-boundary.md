---
type: Observation
title: 'O11: Segment-wise tokenization changed ''\n\n'' into two tokens'
description: The letter prompt differed from the tokenizer's own encoding until boundaries were fixed.
date: '2026-10-04'
confidence: high
tags: []
timestamp: '2026-10-04T00:00:00Z'
---

## Observation

Tokenizing the options block and the instruction separately produced `\n`,`\n` (107,107) where the tokenizer encodes `\n\n` as one token (108). Found by the network test against `apply_chat_template`; fixed by moving the blank line into the instruction segment and stripping state/question whitespace.

## Interpretation

Off-distribution prompts would have handicapped the baseline unfairly.

## Reproduce

`tests/test_formatting.py::test_real_tokenizer_matches_chat_template_and_letters_are_single_tokens` (marker `network`).

## Related

* [d06-letter-logit-baseline](/design/d06-letter-logit-baseline.md)
