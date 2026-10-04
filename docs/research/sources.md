---
type: Reference
title: Sources
description: Sources consulted on 2026-10-03 and the depth of verification for each.
tags: [sources]
timestamp: 2026-10-03T00:00:00Z
---

# Sources

Fetched pages were summarised by a small model, so details are second-hand; vendor claims are unverified.

- [Simon Willison on Jev](https://simonwillison.net/2026/Sep/21/jev/) (fetched)
- [Jev AI: What is System One](https://jevai.net/articles/what-is-system-one-jev/) (fetched, vendor)
- [DataCamp: Jev](https://www.datacamp.com/blog/system-one-models-jev) (fetched)
- [Calibrated Decision Models for Autonomous Pentesting Harnesses (arXiv 2609.28940)](https://arxiv.org/html/2609.28940v1) (fetched)
- Search snippets only, not fetched: [TechTarget](https://www.techtarget.com/it-infrastructure/news/366650696/Jev-decision-model-touted-as-quicker-cheaper-LLM-alternative), [OpenRouter](https://openrouter.ai/blog/insights/what-is-jev/), [Kai Waehner](https://www.kai-waehner.de/blog/2026/09/28/how-system-one-models-like-jev-change-enterprise-ai-architecture/), [Towards Data Science](https://towardsdatascience.com/jev-vs-llms-when-ai-moves-from-generation-to-decision-making/)
- [DFM Mimir v1 (arXiv 2608.13517)](https://arxiv.org/abs/2608.13517), [HF model card](https://huggingface.co/danish-foundation-models/DFM-Mimir) (fetched; abstract and card only)
- [HRM-Text (arXiv 2605.20613)](https://arxiv.org/html/2605.20613v1) (fetched)
- [RLCR: Beyond Binary Rewards (arXiv 2507.16806)](https://arxiv.org/abs/2507.16806) (search snippet only)
- [OKF spec](https://okf.md/spec) (fetched). Note: okf.md's home page says index.md needs `version:`/`entries:` frontmatter, while the spec page says index.md has no frontmatter; this bundle follows the spec page.

## Added 2026-10-04 (all fetched or queried directly unless noted)

- [DFM Mimir v1.5](https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.5): config, model card, `training_data_manifest.json`, `dfm11_sampling_policy.yaml` (revision 521b40b3…)
- [tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions): schema verified by streaming rows (revision d2ab1d12…)
- [bekko-system-one-dataset-v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0): manifests, `sources.json`, parquet rows (revision 5f67e4e2…)
- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): rows inspected (revision d0e2f0c4…); the earlier card summary of its columns was wrong
- [MASSIVE 1.1 archive](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz) and [loading script](https://huggingface.co/datasets/AmazonScience/massive)
- [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) (revision 990b2711…); [ChaosNLI](https://github.com/easonnie/ChaosNLI) (README only, excluded)
- [OKF spec](https://okf.md/spec) section 7 for `log.md`
- Local: transformers 5.13.1 `models/hrm_text` source
