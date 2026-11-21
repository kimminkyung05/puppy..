"""[D] RQ2 단서 변형: 영역(머리/몸/배경), 블러 4단계, 흑백 + 대조군. — 담당 D가 작성

약속:
- 변형마다 이미지를 저장하고 eval_index.csv와 같은 열(image_id, dog_id, path, split)의
  인덱스 csv를 만든다. 예: eval_gray_index.csv
- 특징 추출은 B(딥)·C(SIFT)가 --eval-index 로 받아 subset 이름(eval_gray 등)으로 저장한다.
"""
