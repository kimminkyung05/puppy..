C파트 모델과 예측

checkpoints/   SIFT 혼합 학습 체크포인트 3개 (시드 1·2·3)
               키: state(가중치), classes(127개 순서), mu/sd(표준화 통계), seed, lr, epochs
               사용: Linear(512, 127)에 state를 넣고, 특징을 (x - mu) / sd 로 표준화한 뒤 softmax

predictions/   모델이 출력한 예측 24개
               pred_RQ1-val-*     순종 validation dev 2,540행
               pred_eval_uf*      UF 믹스견 개체 단위 100마리 (dev 20 + hold 80)

CLIP / DINOv2 / DINOv3 체크포인트는 없습니다.
학습 스크립트에 저장 코드가 빠져 예측만 남고 가중치는 보존되지 않았습니다.
재학습 시 장치 차이로 가중치가 달라지는 것을 확인했습니다
(CLIP 시드1 재학습: 1위 예측 일치율 88.2%, top-1 0.7516 -> 0.7598).
공유된 수치와 어긋나므로 재학습하지 않았습니다.

학습 설정: 선형 헤드 1층, AdamW(lr 3e-3, wd 1e-4), 20에폭, 배치 512
           에폭당 혼합 샘플 20,000개(k in [2,4], Dirichlet alpha=1.0) + 순종 전체
           표준화는 학습 데이터 통계만 사용

주의: hold 80마리는 최종 설정 고정 후 1회만 채점하는 규칙입니다.
