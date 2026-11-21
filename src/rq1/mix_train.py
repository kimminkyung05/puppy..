"""[C] RQ1 섞기 학습: 특징을 비율대로 섞은 가짜 믹스 + 소프트 라벨 KL. — 담당 C가 작성

약속:
- 분류기(A)와 같은 구조(선형층 하나), 같은 클래스 순서
- 출력: {preds_root}/pred_RQ1-{backbone}-mix-s{seed}.csv  (dog_id, split, <클래스>)
"""
