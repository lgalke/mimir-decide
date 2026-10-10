# Open questions

One page per open question: why it matters, how to resolve it, linked experiments.

## Question

* [Q01: Where does clean generalisation evidence come from?](/questions/q01-clean-generalisation-evidence.md) - [open] Almost every usable source is flagged as seen by Mimir; we need data it cannot have seen.
* [Q02: Which of our eval rows did Mimir see?](/questions/q02-row-level-overlap-with-mimir-data.md) - [open] Row-level overlap cannot be computed from public artifacts because the DFM10 base data is private.
* [Q03: Does the real 1.8B model train on one GPU?](/questions/q03-real-model-memory-and-speed.md) - [answered] Memory, step time and throughput of Mimir v1.5 under our setup are unmeasured.
* [Q04: Does the slot readout beat the letter baseline on the real model?](/questions/q04-slot-vs-letter-on-real-model.md) - [open] Only a toy synthetic check exists.
* [Q05: Does CC-BY-SA data bind the model weights?](/questions/q05-share-alike-licence-on-weights.md) - [open] Share-alike may or may not extend to a trained model.
* [Q06: How much of bekko survives the licence policy?](/questions/q06-bekko-coverage-after-strict-licences.md) - [open] Most subsets are 'qualified' and excluded; the full-build size is unknown.
* [Q07: Are soft labels probabilities of being correct?](/questions/q07-soft-label-semantics-vs-correctness.md) - [open] Human disagreement and teacher probabilities are not objective correctness probabilities.
* [Q08: Does dropping long states bias the mixture?](/questions/q08-long-state-drop-bias.md) - [open] States over 12,000 characters are dropped; about a quarter of sampled bekko rows were.
* [Q09: Should we add LLM-judge distillation?](/questions/q09-teacher-distillation-stage.md) - [open] Stage 1 of the first plan is not implemented.
* [Q10: Does HRM recurrence help the decision readout?](/questions/q10-does-hrm-recurrence-help-the-readout.md) - [open] Unknown whether more H/L cycles improve decisions or only cost compute.
* [Q11: How does Jev actually work?](/questions/q11-jev-internals.md) - [open] Backbone, size, parallel sampler and RLCD details are not public.
* [Q12: How position-sensitive are the predictions?](/questions/q12-option-position-bias-at-evaluation.md) - [answered] Evaluation uses the given option order; a model could still prefer early or late slots.
* [Q13: Does calibration transfer to held-out sources?](/questions/q13-calibration-transfer-across-domains.md) - [open] Laya's calibration was not shown to transfer across domains.
* [Q14: How much of the fine-tuned baseline's quality comes from fine-tuning?](/questions/q14-how-much-of-the-baseline-quality-is-fine-tuning.md) - [answered] The letter baseline reaches about 0.835 accuracy after fine-tuning. How much would the untrained Mimir already reach with the same prompt?
* [Q15: Is NLL-fitted temperature scaling the right calibration?](/questions/q15-is-nll-fitted-temperature-scaling-the-right-calibration.md) - [open] The fitted temperatures lower NLL slightly but raise ECE on noul and held-out data. Another fitting criterion, or no post-hoc step for the fine-tuned models, may be better.

## Template

* [Template: open question](/questions/template-question.md) - Copy to questions/qNN-slug.md to record an open question.
