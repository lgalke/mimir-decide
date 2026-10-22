---
type: Decision
title: 'D34: Slot and scalar prompts use the Gemma chat format (flag chat_format)'
description: The slot and scalar readouts put their text inside the Gemma user turn followed by the generation prompt, as Mimir saw its training prompts. The letter baseline always did. Slot-v0 and the probes were plain text and keep that format for reproduction.
status: accepted
date: '2026-10-22'
decided_by: owner (the chat template must be kept); implementation by agent
tags: [format, chat-template, distribution]
timestamp: 2026-10-22T00:00:00Z
---

## Context

Mimir was trained on prompts rendered with the Gemma chat template: the prefix of every example is `<bos><|turn>user\n…<turn|>\n<|turn>model\n` and the response follows ([O28](/observations/o28-mimir-basics-check.md)). The letter baseline uses exactly this string (tested against `apply_chat_template`). The slot and scalar prompts were plain text (`State:\n…`), without turn tokens, which is outside the training format. The owner stressed that keeping the template matters because the model otherwise works outside its training distribution. The running 1M letter baseline is not affected.

## Decision

- `Formatter(..., chat=True)` wraps the slot and scalar prompts: `<bos><|turn>user\nState:\n{state}\n\nQuestion ({kind}):\n{question}\n\n{Options:\n[M] a\n[M] b | Option:\n{option}}<turn|>\n<|turn>model\n`. The last option of the slot prompt has no trailing newline, because the template trims user content. The whole sequence stays one bidirectional PrefixLM block.
- **Readouts:** slot unchanged (final state at the markers, inside the user turn); scalar at the last token of the row, which is now the final token of the generation prompt, the position from which Mimir was trained to predict its first response token.
- **Switch:** config key `chat_format` (default `true` for `model_type: scalar`, `false` otherwise); stored in `decision_meta.json` so evaluation, calibration, the Typed Decisions scorer and the Decision Index engine use the format of the checkpoint. `configs/train_slot_v1.yaml` and `configs/train_scalar_v1.yaml` set `chat_format: true`. The letter format is always chat.
- **Reproduction:** `slot-v0`, `slot-probe` and the E13 probes have no `chat_format` in their metadata and are evaluated in the plain format as before.
- Tested token for token against `apply_chat_template` of the real tokenizer (scalar row; slot prompt without the marker tokens, which have no text form), plus order and padding invariance of the scalar scores in the chat format.

## Consequences

- The slot and scalar models trained from now on are in Mimir's prompt format; a slot or scalar model trained in the chat format is not comparable one to one with `slot-v0` (plain). The format is a further difference between v0 and v1 runs, next to the mixture and the context length.
- The chat prefix costs about 12 more tokens per sequence (about 12 per row for the scalar head).
- Not tested: whether the chat format improves quality; a frozen slot probe in chat format might behave differently from [O24](/observations/o24-slot-probe-extracts-little-from-a-frozen-backbone.md).

## Revisit when

A comparison of plain and chat format for the same readout and data shows a difference.

## Related

* [D33](/design/d33-scalar-per-option-head.md)
* [D05](/design/d05-slot-readout-heads.md)
* [D06](/design/d06-letter-logit-baseline.md)
