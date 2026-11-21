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
2. 공식 test split은 보존하고 train에서 validation을 만듭니다.
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
Phase 1 dataset inspection and split validation are complete.

Run `python -m src.data` to regenerate the statistics and split manifest.
The dataset revision is pinned in `config.json`. The statistics record the
runtime versions and the manifest SHA-256. Split checks include the configured
lowercasing and reject overlaps before publishing results. Official test rows
are preserved.

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
