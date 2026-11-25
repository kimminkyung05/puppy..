# D파트 최종 전달본

이 폴더는 팀 공유용 D: RQ2 단서 분석 최종 산출물이다.

## 먼저 볼 파일

1. `D_RQ2_completion_report.md` — 최종 범위와 완료 여부
2. `D_RQ2_FINAL_VERIFICATION.json` — 행 수·seed·TEST/SAM3 검증
3. `D_RQ2_completion_audit.csv` — 항목별 완료/제외 상태
4. `D_DINOv3_extra_drop_by_condition.png` — DINOv3 그래프
5. `SIFT_extra_drop_by_condition.png` — SIFT matched-control 그래프
6. `D_breed_group_definition.csv` — 품종군 정의
7. `D_breed_group_extra_drop_summary.csv`와 `D_breed_group_extra_drop_overlap.svg` — 품종군별 비교
8. `SAM3_RAW_ARTIFACTS_INDEX.md` — 원시 마스크 Drive 위치와 상태 기록

## 최종 범위

- 완료: CLIP·DINOv2·DINOv3·SIFT matched-control, TEST 통합, SAM3 masked 평가
- 제외: background-only, 선택실험/attention, SIFT panel 영역 변형
- CLIP·DINOv2: matched-control extra-drop까지 완료

## 핵심 수치

- DINOv3: manifest 300, 상세 900, extra-drop 상세 420, 요약 189, seed 1·2·3
- SIFT: 입력 220장, control score 1,980, extra-drop 상세 900, 요약 45, seed 1·2·3
- CLIP·DINOv2: raw score 1,800, extra-drop 상세 840, 요약 42, seed 1·2·3
- TEST 통합: 909행
- SAM3: 101개 마스크/상태, body/head masked 결과 각 303행
- `UF089_2`: 매칭 확인 완료
- 품종군 집계: 4,680 detail / 234 summary rows, provisional 3 groups
- 수동 머리 박스 개수: 1장(`UF023`) 확정. `head_boxes.csv`와 `SAM3_mask_status_v2.json` 확인

행 수가 필요한 경우 `D_RQ2_FINAL_VERIFICATION.json`을 기준으로 확인한다.
