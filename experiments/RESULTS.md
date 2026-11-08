# TrustReview experiments (8 Nov 2026)

All runs: DistilBERT-base-uncased, max_len 256, 3 epochs, lr 2e-5, effective batch 32, RTX 3050 6GB.
Model selection is done on a validation split carved from train, using macro F1 across sources. The test set is never used to pick a model.
The v1 test split (10,880 rows, seed 42, stratified by label+source) is byte-identical to the one behind the original 90.88%.

## 1. Re-check of the shipped model
| | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Shipped model, v1 test | 90.88% | 0.912 | 0.978 |
| Per source | Amazon 97.9%, OPUS 85.6%, **Yelp 68.6%** | | |

Note: the original training picked its best epoch on the test set. Retrained with a proper validation split, the same setup scores **90.58%**.

## 2. Why Yelp is low: interventions (Yelp test, 2,474 rows)
| Intervention | Yelp acc | Overall v1 test |
|---|---|---|
| Base (proper val) | 68.1% | 90.58% |
| TF-IDF trained on Yelp only | 68.0% | n/a |
| Source-balanced loss | 68.9% | **91.07%** (OPUS 83.8% to 88.1%) |
| Source tag in input | 69.1% | 90.77% |
| Yelp masked-LM domain adaptation | 67.3% | 90.20% |
| All three text fixes together | 68.6% | 90.06% |
| **Reviewer behaviour features only (GBDT)** | **88.2%** (86.3% on unseen reviewers) | n/a |
| DistilBERT + behaviour late fusion | 88.4% (text weight 0.05 to 0.10) | n/a |

Conclusion: Yelp labels come from Yelp's behaviour-based filter. Text-only models plateau around 68%, in line with Mukherjee et al. 2013. Reviewer metadata carries the signal.

## 3. Cross-domain (leave-one-source-out, scored on all rows of the held-out source)
| Held-out source | TF-IDF | DistilBERT |
|---|---|---|
| Amazon | 49.3% | n/a |
| OPUS | 50.3% | n/a |
| Yelp | 51.9% | 51.0% |

Every held-out source lands at chance. Each source's "fake" is a different phenomenon: GPT-2-era generation (Amazon), crowd-written deception (OPUS), filter-flagged spam (Yelp).

## 4. External test sets (shipped model)
| Dataset | Result |
|---|---|
| GP Surgery reviews, healthcare (real vs DeepSeek-generated), 1,000 | 79.7% acc, 0.862 AUC; flags 19% of real reviews as fake |
| Modern LLM fakes (DeepSeek-v4, Amazon-style) | **catches 1.0%** (AUC 0.44) |

## 5. v2: retrain with modern LLM fakes + healthcare reviews
Added 10K DeepSeek-v4 fakes, 10K LFM2.5 fakes and 9.9K GP Surgery reviews to train, with source-balanced loss.
**GLM-5.2 is kept out of training entirely** and appears in test only, as an unseen generator.
Any train row found in a val/test set is removed (0 train/test overlap).

| Test slice | Base (v1 data) | **v2** |
|---|---|---|
| Full v2 test (24,473) | 60.7% | **93.2%** (AUC 0.988) |
| DeepSeek-v4 fakes caught | 1.3% | 99.7% |
| LFM2.5 fakes caught | 1.5% | 99.7% |
| **GLM-5.2 fakes caught (unseen generator)** | 0.7% | **98.3%** (AUC 0.994) |
| GP Surgery | 78.5% | 99.9% |
| Amazon (v1) | 97.7% | 95.5% |
| OPUS (v1) | 83.8% | 85.0% |
| Yelp (v1) | 68.1% | 67.3% |
| v1 test overall | 90.58% | 88.8% |

Caveats:
- GLM fakes were generated with the same grounding prompt as the DeepSeek and LFM fakes. "Unseen generator" here means a new model under the same protocol, not a new protocol.
- GP fakes appear easy to separate (99.9%), likely from templated generation.
- v2 gives up about 2 points on the old Amazon test, mostly because more genuine reviews get flagged as fake.

## Reproduce
```
python classical.py                 # sections 2 (behaviour), 3 (TF-IDF LOSO)
bash run_all.sh                     # section 2 DistilBERT runs + LOSO + fusion
python alien.py <gp_test.csv> <modern_test.parquet>   # section 4
bash run_v2.sh                      # section 5
```
Datasets: `TRUST_REVIEW_DATASETS_DIR` (Amazon/OPUS/Yelp), `TRUST_REVIEW_ALIEN_DIR` (HF: Flowerly/modern-fake-reviews{,-lfm,-glm}, janduplessis886/gp_surgery_reviews_fake_and_real).
