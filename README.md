# DNLP Prompt Injection Detection

딥러닝 자연어처리(DNLP) 과목 프로젝트입니다.

## Setup

Python 3.14.0에서 검증했습니다. Windows PowerShell에서 다음을 실행합니다.

```powershell
git clone https://github.com/rty7683-rgb/dnlp-prompt-injection.git
cd dnlp-prompt-injection
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce

저장소 루트에서 아래 순서로 실행합니다.

```powershell
python -m src.data
python -m src.vocab
python -m src.train
python -m src.predict
```

1. `src.data`는 고정 revision의 데이터셋을 다운로드하고 검증한 뒤 train/validation/test의 고정 membership을 재현합니다.
2. `src.vocab`은 train에서만 vocabulary를 만들고 validation OOV 통계와 token→ID 예시를 생성합니다.
3. `src.train`은 고정된 train/validation 분할로 A/B/C validation 비교를 재현합니다. 학습에는 train만 사용합니다.
4. `src.predict`는 저장된 최종 Model B를 로드해 새 입력 5개를 예측합니다. 모델 학습은 수행하지 않습니다.

`src/final_evaluation.py`는 동결된 A/B/C의 공식 test 평가를 담당합니다.
공식 평가는 이미 완료됐으며, `python -m src.final_evaluation`은 기존
결과 파일이 있으면 중단합니다. 제출된 결과를 확인하려면 다음을 실행합니다.

```powershell
python -m src.final_evaluation --verify-existing
```

이 모드는 동결 설정으로 train 432행에서 A/B/C를 메모리에서 재구성하고,
공식 test 116행의 결과를 저장된 지표 및 예측과 비교합니다. 저장된 Model B와
특징 매핑도 검증합니다. 지표와 예측 라벨은 정확히 일치해야 하며 로그 점수는
부동소수점 차이를 고려해 절대 오차 `1e-10`까지 허용합니다. 파일은 쓰지 않습니다.

`src/error_analysis.py`는 저장된 test 예측과 최종 Model B의 validation
예측을 사용해 수동 검토 사례, 점수, OOV 통계를 내보냅니다.
Phase 1의 `near_duplicate_audit.csv`는 수동 판정이 행 순서를 참조하므로
동결되어 있습니다. 수동 판정 파일이 있으면 audit 스크립트가 덮어쓰기를 막습니다.

## Data and test policy

`deepset/prompt-injections`의 revision은 `config.json`에 고정했습니다.
공식 test 116행을 모두 보존했습니다. 학습 전 누수 점검에서 공식 test 질문의
변형으로 판정된 공식 train 5행은 개발 데이터에서 제외했습니다. 최종 분할은
train 432행, validation 109행, test 116행입니다. Validation으로 Model B를
선택하고 설정을 동결한 뒤 공식 test에서 최종 평가했으며, 결과 확인 후 설정 변경은 없었습니다.

## Main artifacts

| 파일 | 내용 |
|---|---|
| [artifacts/vocab.json](artifacts/vocab.json) | Train에서 구축한 token→ID 매핑 |
| [artifacts/final/model_b.npz](artifacts/final/model_b.npz) | 동결된 최종 Model B |
| [artifacts/final/features_b.json](artifacts/final/features_b.json) | Model B의 unigram 특징 매핑 |
| [results/data_stats.json](results/data_stats.json) | 데이터·분할 통계와 manifest 해시 |
| [results/token_id_examples.json](results/token_id_examples.json) | 정규화·토큰화·ID 변환 예시 |
| [results/vocab_stats.json](results/vocab_stats.json) | Vocabulary 크기와 validation OOV |
| [results/validation_experiments.json](results/validation_experiments.json) | A/B/C validation 지표 |
| [results/test_experiments.json](results/test_experiments.json) | 동결 모델의 공식 test 지표 |
| [results/test_predictions.csv](results/test_predictions.csv) | 공식 test의 Model B 예측과 점수 |
| [results/error_analysis.csv](results/error_analysis.csv) | Test 7건과 validation 3건의 사례 분석 |
| [results/new_inputs.json](results/new_inputs.json) | 새 입력 5개의 예측과 참고 라벨 |

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

### Phase 6 new-input prediction

Run `python -m src.predict` to reproduce the five examples using the saved
Model B. Four predictions matched the reference labels, covering a clear
benign question, a direct injection, an indirect injection, and a
spelling-variant injection. The model labeled a benign question about the
meaning of “ignore previous instructions” as an injection. This illustrates
a limitation of unigram Bag-of-Words: it captures which words appear, but
cannot reliably tell whether an expression is a command or is being quoted
for explanation.

The demo loads the frozen Model B and its saved feature mapping. The
reference labels are used only to compare predictions. No fitting or
dataset loading occurred, and the model and settings were preserved.

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
