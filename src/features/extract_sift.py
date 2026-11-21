"""[C] SIFT + BoVW 특징 추출 (CPU). — 담당 C가 작성

약속 (B·A·D가 그대로 읽을 수 있게):
- 저장 형식은 src/common/features_io.py 와 같다. backbone 이름은 'sift'.
- cls, mean 모두 BoVW 히스토그램(정규화)을 넣는다.
- 학습셋은 tsinghua.load_crop()의 몸통 박스 크롭, 평가셋은 eval_index.csv의 path를 쓴다.
- 코드북(k-means)은 학습셋 기술자 일부로 만들고 outputs/에 저장해 재사용한다.
"""
