# D: RQ2 단서 분석 최종 완료 보고서

## 결론

최종 합의된 범위에서는 D파트를 완료했다. 최종 범위는 CLIP·DINOv2·DINOv3·SIFT의 matched-control 변형 실험, 새 TEST셋 통합, SAM3 개/머리 마스크 기반 평가다.

원래 초안에 있던 배경-only 실험, 선택실험(어텐션 그림 포함)은 진행하지 않기로 한 결정에 따라 제외했다. 따라서 이 패키지에서 `excluded`는 누락이 아니라 최종 범위에서 의도적으로 제외된 항목이다.

## 완료된 산출물

- DINOv3: `orig / gray / blur1 / blur3 / blur6 / blur12`와 `panel_face / panel_body`, A 분류기 seed 1·2·3.
  - manifest 300행, 상세 900행, extra-drop 상세 420행, 요약 189행.
- SIFT: C파트의 `sift_codebook.npy`와 SIFT 분류기 seed 1·2·3을 사용해 `gray / blur1 / blur3 / blur6 / blur12`와 PSNR matched-control을 재실행.
  - 20개 평가 이미지, 입력 220장, control score 1,980행, extra-drop 상세 900행, 요약 45행.
  - 흑백 sanity check는 PASS.
- CLIP·DINOv2: DINOv3와 동일한 300장 변형/대조군 이미지에 대해 seed 1·2·3을 재채점.
  - 20개 평가 이미지, raw score 1,800행, extra-drop 상세 840행, 요약 42행.
- 새 TEST셋: CLIP·DINOv2·DINOv3 통합 결과 909행, 요약 18행.
- SAM3: 101개 TEST 이미지의 마스크/상태와 body/head masked DINOv3 결과 각 303행.
- `UF089_2` 매칭은 재확인 결과 `matched=True`.
- 품종군별 집계: 네 추출기의 extra-drop 상세 4,680행, 요약 234행, 그래프 1개. 품종군은 털 유형 기반 provisional 3개 군이다.
- 수동 머리 박스: `UF023.jpg`를 확인해 좌측 face panel에 수동 박스를 지정했다. 총 1장이고 `head_boxes.csv`, `SAM3_mask_index_v2.csv`, `SAM3_mask_status_v2.json`에 기록했다. 기존 masked-evaluation CSV는 이 annotation 이전 실행 결과이므로 재실행 전까지 그 상태를 명시한다.

## 해석 범위

이 패키지의 CLIP·DINOv2·DINOv3·SIFT 표·그래프가 최종 단서 분석 결과다. 네 추출기 모두 같은 matched-control 정의를 사용했으며, SIFT는 패널 영역 변형을 제외한 gray/blur 범위로 계산했다.

## 제외 항목

- background-only 변형
- 선택실험
- 선택실험에 해당하는 attention 그림
- SIFT의 panel_face/panel_body 영역 변형

상태와 행 수의 근거는 `D_RQ2_FINAL_VERIFICATION.json`과 `D_RQ2_completion_audit.csv`에 기록했다.
