# Changes and implementation log

이 문서는 과제에서 요구하는 **직접 구현·수정한 부분, 외부 코드/라이브러리, AI 사용 내역**을 기록합니다.

## Direct implementation
현재는 프로젝트 기본 구조만 생성한 상태입니다.

예정된 직접 구현 항목:
- text normalization / word tokenizer
- train-only vocabulary builder
- token-to-ID conversion and UNK handling
- validation OOV-rate calculation
- unigram / bigram feature extraction
- majority-class baseline
- Multinomial Naive Bayes
- Laplace smoothing
- model save/load
- prediction for new inputs
- error-analysis export

## External libraries
- `datasets`: 공개 데이터셋 다운로드
- `numpy`: 수치 계산
- `pandas`: 데이터 정리 및 결과 저장
- `scikit-learn`: 데이터 split 및 평가 지표
- `matplotlib`: 결과 시각화

## AI usage
- ChatGPT: 프로젝트 주제 검토, 실험 설계, 저장소 기본 구조 설계 및 초기 scaffold 생성 지원.
- 이후 코드 작성/디버깅에 AI 도움을 사용한 경우 해당 파일과 내용을 계속 기록합니다.

## Change log
### Phase 1 dataset inspection
- Codex assisted with fixing `src/data.py` to read the normalized text by position when generating manifest hashes.
- Ran `python -m src.data` and verified the 662 manifest rows against the split counts and label distributions in `results/data_stats.json`. The split sizes are 436 train, 110 validation, and 116 test.
- Found no blank rows, conflicting labels, or exact duplicates after duplicate-check normalization. The official test retained all 116 rows.

### Initial scaffold
- 프로젝트 주제를 prompt injection detection으로 확정
- A/B/C 비교 실험 구조 설계
- 기본 디렉터리와 설정/문서 파일 추가
