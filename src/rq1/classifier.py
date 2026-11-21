"""[A] RQ1 분류기: 특징 위 선형층, 127클래스 분류 + dev 온도 조정. — 담당 A가 작성

약속:
- 입력: features_io.load_features(root, backbone, 'train'/'val'/'eval')
- 클래스 순서: 학습셋 label의 정렬 순서 (prototype_fcls.py와 같음)
- 출력: {preds_root}/pred_RQ1-{backbone}-clf-s{seed}.csv  (dog_id, split, <클래스>)
- 체크포인트: D가 RQ2 채점에 쓴다.
"""
