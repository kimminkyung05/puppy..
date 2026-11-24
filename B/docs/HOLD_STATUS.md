# B파트 hold 평가 상태

**B는 hold를 아직 채점하지 않았다.** 모든 B 결과는 dev(UF 20마리, Tsinghua val dev 절반)다.

## hold 전에 고정할 조건

- 추출기: CLIP · DINOv2 · DINOv3, `cls` 특징 (`configs/PREPROCESSING.md`)
- 평균점: 학습셋 품종별 평균, 반복 r1~r3마다 품종별 80% 부분 표본
- L2 정규화: 없음 (dev에서 차이 0.2%p 이하. 팀 확인 필요)
- 믹스견 집계: face · body 사진의 비율을 개 단위로 평균, 반복 3회의 확률 평균을 대표값으로 쓴다
- 지표: `docs/METRICS_FINAL.md` (A 최종 정의 복사본). 주 지표는 원래 DNA 비율 포착 혈통 비율@3 + coverage, 보조는 재정규화 포착@3 · 재정규화 분포 겹침 · Top-1 + coverage
- 품종 매핑: `A/configs/breed_map.csv`의 `status=정확`만

## 한 번만 채점

팀이 위 조건을 최종으로 확인한 뒤 hold를 한 번만 채점한다. 결과를 본 뒤 설정 · 매핑 · 제외 기준을 바꾸지 않는다.
A는 2026-11-23에 hold를 한 번 채점했다(`A/results/HOLD_EVALUATED_ONCE.json`).
