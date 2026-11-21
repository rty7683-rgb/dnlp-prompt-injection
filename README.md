# DNLP Prompt Injection Detection

딥러닝 자연어처리(DNLP) 과목 프로젝트입니다.

## Project goal
사용자의 프롬프트를 **정상(normal)** 또는 **프롬프트 인젝션(prompt injection)** 으로 분류합니다.

- Dataset: `deepset/prompt-injections`
- Baseline A: Majority Classifier
- Model B: Word Unigram + Bag-of-Words + Multinomial Naive Bayes
- Model C: Word Unigram+Bigram + Bag-of-Words + Multinomial Naive Bayes
- Main metrics: Accuracy, Macro-F1
- One-factor comparison: n-gram range (B: 1 / C: 1~2)

## Research question
> Word Unigram만 사용할 때보다 Bigram을 함께 사용할 때 프롬프트 인젝션 탐지 성능이 개선되는가?

## Planned workflow
1. 데이터셋 구조·라벨 분포·빈 문장·중복을 점검합니다.
2. 공식 test 116행을 보존합니다. 수동 누수 점검에서 확인한 test 변형 5행을 공식 train에서 제외하고, 나머지 541행을 변형 그룹 단위로 train과 validation으로 나눕니다.
3. 텍스트 정규화와 Word Tokenizer를 구현합니다.
4. **train에서만** vocabulary를 만들고 validation OOV rate를 계산합니다.
5. Majority baseline을 구현합니다.
6. Laplace smoothing을 포함한 Multinomial Naive Bayes를 직접 구현합니다.
7. Unigram(B)과 Unigram+Bigram(C)을 같은 조건에서 비교합니다.
8. 설정을 확정한 뒤 공식 test에서 최종 평가합니다.
9. 오류 사례 약 10개와 직접 만든 새 입력 5개를 분석합니다.
10. README, changes.md, 결과 파일과 보고서를 정리합니다.

## Repository structure
```text
.
├── src/
│   ├── models/
│   ├── data.py
│   ├── preprocessing.py
│   ├── tokenizer.py
│   ├── vocab.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── error_analysis.py
├── artifacts/
├── results/
├── report/figures/
├── config.json
├── requirements.txt
├── changes.md
└── .gitignore
```

## Reproducibility policy
- Random seed는 `config.json`에서 고정합니다.
- Vocabulary와 전처리 통계는 train split에서만 학습합니다.
- Validation은 설정 선택에만 사용합니다.
- Test 결과를 확인한 뒤 설정을 다시 변경하지 않습니다.
- 원본 데이터 전체, 가상환경, 캐시는 저장소에 커밋하지 않습니다.

## Current status
Phase 1 split validation and Phase 2 preprocessing, word tokenization, and
train-only vocabulary construction are complete. Phase 3 validation is
complete. Phase 4 official test evaluation is complete, and the final model
remains Model B, as selected on validation. Official test results were used
only for final evaluation.

| Model | Official test Accuracy | Official test Macro-F1 |
|---|---:|---:|
| A · Majority | 0.4828 | 0.3256 |
| B · Unigram + Multinomial Naive Bayes | 0.9397 | 0.9397 |
| C · Unigram + Bigram + Multinomial Naive Bayes | 0.9310 | 0.9310 |

Model B's confusion matrix is `[[55, 1], [6, 54]]`, with true labels on rows
and predicted labels on columns in order `[0, 1]`.

| Model B class | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 | 0.9016 | 0.9821 | 0.9402 |
| 1 | 0.9818 | 0.9000 | 0.9391 |

All 116 predictions matched after reloading the final Model B artifact and
feature mapping.

### Phase 5 error analysis

We reviewed all seven official test errors (1 FP, 6 FN) and three representative
validation errors. Recurring patterns included instructions appended to benign
questions or search phrases, language switches, and indirect instructions.
OOV rates alone did not explain the errors. Model C also had six official test
false negatives. Case-level overlap with Model B was not checked.

Run `python -m src.train` to reproduce the fixed validation comparison.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| A · Majority | 0.6239 | 0.3842 |
| B · Unigram + Multinomial Naive Bayes | 0.9174 | 0.9124 |
| C · Unigram + Bigram + Multinomial Naive Bayes | 0.8991 | 0.8930 |

B led on both metrics and is frozen with the Phase 2 preprocessing and
tokenizer, unigram counts, the fixed split, `min_freq=1`, and `alpha=1`.
Adding bigrams did not improve validation performance. The feature space
grew from 2,057 to 7,116 with 432 training examples. Greater feature sparsity
is a possible explanation to investigate during error analysis.
The custom Naive Bayes matched sklearn's predictions and log probabilities
on toy data. Phase 3 used only train and validation.

Run `python -m src.vocab` to build the word vocabulary and generate validation
OOV statistics and token ID examples. It requests only the official train
split and uses the verified manifest to recover train and validation.

The final vocabulary uses `min_freq=1`. Its 2,058 entries (including
`<UNK>`) retain all 9,435 train tokens without mapping any to `<UNK>`.
For this small dataset, we chose to retain rare tokens. Validation OOV is
526 of 2,955 tokens (17.8003%), compared with 25.4822% at `min_freq=2`
and 31.0998% at `min_freq=3`.

Run `python -m src.data` to regenerate the statistics and split manifest.
The dataset revision is pinned in `config.json`. The statistics record the
runtime versions and the manifest SHA-256. Split checks include the configured
lowercasing and reject overlaps before publishing results. Official test rows
are preserved. The original candidate audit is frozen so that the one-based
pair indices in `results/near_duplicate_review.csv` continue to identify the
same pairs. Reviewed variants form connected families. A family that touches
official test is excluded from development data on the official-train side.

With seed 42, five-fold `StratifiedGroupKFold` keeps each remaining family
together. The validation fold minimizes squared differences from the target
per-label counts for the 20% holdout. The final train, validation, and test
sizes are 432, 109, and 116.

Use `src.data.load_verified_splits()` to reuse the saved train and validation
membership. Pass `include_test=True` for final evaluation after the settings
have been fixed. The loader verifies the saved membership, labels, and hashes.

## Dataset
- Hugging Face dataset: `deepset/prompt-injections`

원본 데이터는 저장소에 포함하지 않으며 코드에서 지정된 revision을 다운로드합니다.

## Environment
Phase 1 검증 환경은 Python 3.14.0이며 직접 사용하는 패키지의 버전은
`requirements.txt`에 고정되어 있습니다.

```bash
pip install -r requirements.txt
```
