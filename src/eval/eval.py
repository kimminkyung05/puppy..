"""[A] 채점: 예측 CSV(클래스 공간) → breed_map.csv로 DNA 품종 공간에 합침 → 지표. — 담당 A가 작성

약속:
- 입력: pred_*.csv (dog_id, split, <Tsinghua 클래스 127개>), 정답 csv, breed_map.csv
- 주 지표: 포착 혈통 비율(상위 3품종의 실제 DNA 비율 합)
- 보조: 분포 겹침, Top-1, 구간별 재현율, KL, 층화, 부트스트랩 95% CI
"""
