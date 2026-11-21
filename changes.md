# Changes and implementation log

이 문서는 과제에서 요구하는 **직접 구현·수정한 부분, 외부 코드/라이브러리, AI 사용 내역**을 기록합니다.

## Direct implementation

- Phase 1 - Dataset inspection, leakage and near-duplicate audit support, and reproducible split validation.
- Phase 2 - Text normalization, word tokenization, train-only vocabulary, token-to-ID and UNK handling, and validation OOV measurement.
- Phase 3 - Unigram/bigram BoW extraction, majority baseline, custom Multinomial Naive Bayes, Laplace smoothing, model save/load, and evaluation.
- Phase 4 - Frozen official test evaluation, prediction export, and read-only verification of saved results.
- Phase 5 - Frozen model error analysis and export.
- Phase 6 - Saved final model prediction for new inputs.

## External libraries
- `datasets`: 공개 데이터셋 다운로드
- `numpy`: 수치 계산
- `pandas`: 데이터 정리 및 결과 저장
- `scikit-learn`: 데이터 split 및 평가 지표
- `matplotlib`: 결과 시각화

## Team / Roles

This is an individual project.

- Student - Topic selection, final experiment design decisions, experiment execution and review, configuration decisions, result interpretation, report writing, and submission.

## AI usage

- ChatGPT - Compared topic candidates, reviewed the experiment design and submission requirements, cross-checked results at each stage, helped plan error analysis, and helped design the new-input examples.
- Codex CLI - Assisted with implementation, debugging, test execution, reproducibility checks, and updates to README.md and changes.md.
- The student finalized model settings and experiment choices based on the assignment requirements and validation results. Model settings were frozen before official test evaluation and remained fixed afterward.

## Change log
### Submission cleanup
- Consolidated setup, reproduction commands, artifact links, individual roles, and AI usage. Added read-only verification of the stored official test results and saved Model B.

### Phase 6 new-input prediction
- Codex added a prediction entry point using the saved final Model B and unigram mapping. Ran five demonstration inputs designed with ChatGPT assistance and accepted for the project and recorded normalized log probabilities, margins, unseen unigram counts, and agreement with the reference labels in `results/new_inputs.json`.
- Four of five predictions matched the reference labels. Verified reproducible output with model fitting and dataset loading disabled.

### Phase 5 error analysis
- Codex exported all seven official test errors from saved Model B predictions and reviewed three validation errors using the frozen artifact. The CSV records source text, scores, margins, OOV counts, and case-specific hypotheses. No model was fit or changed.

### Phase 4 official test evaluation
- Codex implemented and ran the frozen A/B/C evaluation once on all 116 official test rows. Each model was fit on the same 432 train rows used in Phase 3.
- Recorded test metrics and Model B predictions by source index, and saved the train-only Model B and feature mapping. Reloaded predictions matched exactly. Model B remained the validation-selected model.
- Official test Accuracy / Macro-F1 were A 0.4828 / 0.3256, B 0.9397 / 0.9397, and C 0.9310 / 0.9310. Test results were used only for final evaluation.
- Model B's confusion matrix was `[[55, 1], [6, 54]]` in label order `[0, 1]` (actual rows, predicted columns). Precision / Recall / F1 were 0.9016 / 0.9821 / 0.9402 for class 0 and 0.9818 / 0.9000 / 0.9391 for class 1.

### Phase 3 validation experiments
- Codex assisted with implementing train-only word n-gram count features, the majority baseline, and log-space Multinomial Naive Bayes with smoothing and model serialization.
- Run `python -m src.train` for A/B/C metrics on the fixed validation split. Models B and C share preprocessing, tokenizer, split, classifier, and alpha. C adds bigrams to B's unigram features.
- Validation n-grams outside the train feature vocabulary are ignored. The Phase 2 UNK ID is used for OOV reporting and is excluded from classifier features.
- Final validation Accuracy / Macro-F1 were A 0.6239 / 0.3842, B 0.9174 / 0.9124, and C 0.8991 / 0.8930. Froze B because it ranked first on both metrics. B uses unigram counts and alpha 1 with the fixed Phase 1 split and Phase 2 settings.
- Adding bigrams did not improve validation performance. The feature vocabulary grew from 2,057 to 7,116 with 432 training rows. Greater feature sparsity is a hypothesis for later error analysis.
- Verified per-class metrics and confusion matrices, shared B/C unigram counts and one-factor conditions, and toy prediction/log-probability parity with sklearn. The official test split was not loaded or evaluated, and the Phase 1 manifest and Phase 2 vocabulary were unchanged.

### Phase 2 preprocessing and vocabulary
- Codex assisted with implementing NFKC, whitespace normalization, configured lowercasing, and a deterministic Unicode word tokenizer that preserves punctuation and internal apostrophes and hyphens.
- Built the vocabulary from train frequencies only, with frequency-descending and token-ascending ordering and `<UNK>` at ID 0. Exported validation OOV statistics and three token ID examples.
- Finalized `min_freq=1` after comparing thresholds 1–3 on the fixed development split. The resulting 2,058-entry vocabulary retains all 9,435 train tokens, avoiding rare-token loss in the small dataset. Validation OOV is 526 of 2,955 tokens (17.8003%), versus 25.4822% and 31.0998% at thresholds 2 and 3.
- Updated the default verified loader to request only the official train split. Verified byte-identical output on rerun, train-only vocabulary frequencies, validation encoding, and preservation of the Phase 1 manifest and audit files.

### Phase 1 final split
- Codex recorded the 19 manual decisions as one-based audit pair indices and decisions, with no raw text.
- Excluded official-train indices 176, 235, 284, 356, and 206 based on the pre-training leakage review. Preserved all 116 official test rows.
- Kept all nine internal variant families intact using a five-fold stratified group split with seed 42. Selected the fold closest to the target per-label validation counts, yielding 432 train and 109 validation rows.
- Verified zero reviewed cross-split variant leakage and byte-identical regeneration of split artifacts.

### Phase 1 near-duplicate audit
- Codex implemented the audit in `src/near_duplicate_audit.py`. The Phase 1 audit CSV is intentionally frozen because the manual review references its row order. The script refuses to overwrite the audit after review.
- Compare lowercase normalized text across split pairs. Flag character 3–5-gram hashing cosine of at least 0.85, or token-set Jaccard of at least 0.75 with at least five shared tokens.
- Record 19 candidate pairs in `results/near_duplicate_audit.csv` (12 train–validation, 6 train–test, 1 validation–test). The CSV stores only split names, original row indices, and similarity scores. Existing split membership was preserved.

### Phase 1 review follow-up
- Codex updated the Phase 1 data pipeline, configuration, dependency pins, and README.
- Preserve every official test row and reject blank or duplicate test rows, invalid binary labels, and overlaps across splits before publishing results.
- Pin the dataset revision, record runtime versions, stage both results before publishing, and verify the manifest checksum and source membership when loading saved splits.
- Re-ran Phase 1 on the pinned revision and verified that the train, validation, and official test membership stayed the same.

### Phase 1 dataset inspection
- Codex assisted with fixing `src/data.py` to read the normalized text by position when generating manifest hashes.
- Ran `python -m src.data` and verified the 662 manifest rows against the split counts and label distributions in `results/data_stats.json`. The split sizes are 436 train, 110 validation, and 116 test.
- Found no blank rows, conflicting labels, or exact duplicates after duplicate-check normalization. The official test retained all 116 rows.

### Initial scaffold
- 프로젝트 주제를 prompt injection detection으로 확정
- A/B/C 비교 실험 구조 설계
- 기본 디렉터리와 설정/문서 파일 추가
