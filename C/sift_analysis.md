SIFT 성능 분석
SIFT+BoVW가 왜 낮게 나오는지 선행 연구와 대조해 정리한다. 두 가지를 분리해야 한다. 순종 분류의 낮은 성능은 정상이고, 믹스견에서 찍기 아래로 떨어지는 것은 별개의 파이프라인 문제다.

1. 순종 Top-1 9.5%는 문헌 범위 안이다
연구	데이터	방법	Top-1
CS231A 프로젝트 (2016)	Columbia Dogs 133종	회색조 SIFT BoW	~5%
본 연구	Tsinghua 127종	회색조 SIFT BoVW, K=512	9.5%
ImageNet 2012 FGVC 엔트리	Stanford Dogs 120종	색 이름(color names) 단독	6.2%
같은 논문	같음	opponent-SIFT BoW (색 포함)	21.6%
Tsinghua 공식 벤치마크	Tsinghua 130종	Inception v3	77.7%
같음	같음	WS-DAN (Inception v3)	86.4%
같음	같음	TBMSL-Net (ResNet50)	83.7%
회색조 SIFT가 들어갈 자리에 정확히 들어가 있다. 동시에 같은 파이프라인의 딥 특징(CLIP 72.5%, DINOv2 85.6%, DINOv3 87.9%)도 Tsinghua 공식 벤치마크 대역(77.7~86.4%)에 들어간다. 양쪽 모두 문헌과 맞으므로 순종 쪽 구현은 정상이다.

Return of the Devil in the Details는 얕은 인코딩과 CNN의 격차를 PASCAL VOC dog 클래스에서 약 23 AP로 보고한다(얕은 IFV 56.6 대 DeCAF 79.5). 품종처럼 세밀한 과제에서는 격차가 더 벌어지며, 본 연구에서는 78%p다.

2. 왜 낮은가: 네 가지 메커니즘
2.1 색을 버린다 (가장 큰 요인)
BoVW 추출 전에 흑백으로 변환하므로 털색 정보가 사라진다. 문헌에서 색 이름만으로 6.2%가 나오고, 같은 BoW 틀에 색을 넣은 opponent-SIFT는 21.6%까지 올라간다. 색 단독이 우리 회색조 SIFT(9.5%)와 비슷한 수준이라는 점이, 색이 품종 정보의 큰 축임을 보여 준다.

2.2 공간 배치를 버린다
BoVW는 "어떤 패턴이 몇 개 있나"만 센다. 품종은 "어떤 패턴이 어디에 어떤 비율로 있나"로 갈린다. 귀가 머리 위에 섰는지 옆으로 처졌는지, 주둥이가 머리 길이의 몇 할인지는 히스토그램에서 사라진다.

2.3 SIFT 기술자가 품종 차이에 둔감하다
SIFT는 조명·스케일 불변을 얻으려고 그래디언트 방향 히스토그램을 정규화한다. 그 과정에서 털 길이나 주둥이 비율 같은 미세 차이가 뭉개진다. 반면 배경 질감(풀, 벽돌, 울타리)은 선명하게 잡혀 단어 분포를 지배한다.

관측으로도 확인된다. UF 100마리에서 SIFT의 1위 예측은 질감이 두드러진 품종으로 쏠렸다.

SIFT	DINOv2
자주 나온 1위	Airedale 13, French_bulldog 6, Saluki 5	Staffordshire 14, Labrador 11, Rhodesian_ridgeback 10
3위 안에 자주 든 품종	Saluki 19, Airedale 16, Pomeranian 13	Staffordshire 24, Labrador 17, AmStaff 17
Airedale은 철사털, Saluki는 장식털, Pomeranian은 솜털이다. UF 개들이 대부분 핏불·래브라도 계열 보호소 믹스견이라는 점과 맞지 않는다.

2.4 코드북 512는 작다
세밀 분류에서는 보통 수천에서 수만 개의 시각 단어를 쓴다. 127종을 512개 단어로 나누면 단어당 정보량이 부족하다.

2.5 과제 자체의 난이도
Khosla 등은 Stanford Dogs의 난점으로 낮은 클래스 간 변이와 높은 클래스 내 변이를 든다. 바셋하운드와 블러드하운드는 얼굴이 비슷하고 색만 다르며, 같은 품종 안에서도 털색이 일정하지 않다. 색을 버린 표현에 특히 불리한 조건이다.

3. 믹스견 붕괴는 표현력 문제가 아니다
표현력이 약하기만 하면 성능은 찍기 근처로 수렴한다. 찍기 아래로는 내려가지 않는다. 그런데 포착 혈통 비율에서 SIFT는 분류기 0.017, 섞기 학습 0.037이고 찍기는 0.088이다. 문헌의 어떤 얕은 방법도 이런 패턴을 보이지 않는다.

원인: 학습 특징과 평가 특징의 코드북 공간 불일치
원본 추출 셀이 저장된 코드북을 그대로 쓰지 않고 한 번 더 k-means를 돌렸다.

km = MiniBatchKMeans(n_clusters=K, init=centers, n_init=1, max_iter=1).fit(centers)
random_state가 없어 실행마다 중심점이 달라진다. 학습 특징이 실제로 사용한 단어 공간은 저장된 sift_codebook.npy와 다르며, 재현이 불가능하다.

증거
값
저장된 코드북으로 학습 이미지 재추출 시 원본과의 코사인
중앙값 0.662, 최소 0.484
학습 특징에서 한 번도 사용되지 않은 시각 단어
73 / 512
이미지별 |z| 최댓값 중앙 (Tsinghua val)
2.6
이미지별 |z| 최댓값 중앙 (UF)
약 133,000
학습 데이터 기준 표준편차 < 1e-3 인 차원
73 / 512
죽은 차원 73개에 UF 특징은 값이 들어가므로 z값이 폭주하고, 소수 차원이 로짓을 지배해 특정 품종으로 쏠린다.

크롭 가설은 기각되었다. 같은 Tsinghua val 사진을 학습과 동일하게 몸 박스로 크롭해도 현재 방식으로 추출하면 top-1이 0.003으로, 전체 이미지(0.003)와 같다. 입력 형태가 아니라 특징 공간이 문제다.

영향 범위
무효: SIFT UF 결과 전부 (C의 섞기 학습, A의 분류기 모두 해당)
유효: 순종 dev 결과, 딥 특징(CLIP/DINOv2/DINOv3) 전체
재사용 가능: D에게 전달한 sift_eval_uf_v5_all.npy는 저장된 코드북으로 직접 배정했으므로, 학습 특징만 같은 방식으로 다시 만들면 맞는다
수정 후 검증
저장된 코드북에 직접 최근접 배정하는 방식으로 학습 특징을 다시 추출하고 (sift_all_v3.npy, 미사용 시각 단어 73 → 0), 동일 설정으로 SIFT 헤드를 시드 3개 재학습해 UF dev 20마리를 다시 채점했다.

지표
v2 (오류)
v3 (수정)
찍기
상한
포착@3
0.0375
0.0326
0.0879
0.5412
재정규화 포착@3
0.0630
0.0621
0.1682
0.970
분포 겹침
0.0167
0.0711
0.0469
1.000
Top-1
0/18
2/18
0/18
—
v3 시드별 포착@3: 0.0403, 0.0701, 0.0451 (편차가 크다) 순종 validation dev Top-1 은 0.0756 / 0.0705 / 0.0791 로 v2(0.0774 평균)와 같은 수준이다. 학습과 평가가 같은 공간이었던 순종 쪽은 수정의 영향을 받지 않는다.

1위 예측 품종의 변화

버전
자주 뽑힌 1위
v2 (오류)
Airedale, briard, Saluki, Pomeranian — 철사털·장식털·솜털
v3 (수정)
golden_retriever 8, Labrador_retriever 6
UF 개체는 대부분 보호소 믹스견이므로 리트리버 계열이 흔하다. 질감 품종으로의 병적 쏠림이 사라졌고, 이는 특징 공간 불일치가 쏠림의 원인이었음을 보여 준다.

결론

수정은 두 가지를 바꿨고 한 가지는 바꾸지 못했다.

예측 분포가 정상화되었다 (질감 품종 쏠림 제거)
분포 겹침이 4배 올라 찍기 기준선을 넘었다
그러나 포착@3 은 개선되지 않았고 여전히 찍기의 절반 수준이다
v3 모델은 20마리 중 14마리에 리트리버 둘 중 하나를 1위로 부여한다. 비율의 크기는 어느 정도 맞으므로 겹침은 올라가지만, 개체마다 다른 품종을 골라내지 못하므로 포착@3 은 그대로다.

따라서 SIFT 의 낮은 믹스견 성능은 구현 오류와 표현 한계가 함께 작용한 결과이며, 오류를 제거한 뒤에도 표현 한계는 남는다. 2절에서 정리한 메커니즘 (색·공간 정보 폐기, 기술자의 둔감성, 작은 코드북)이 그 한계의 내용이다.

본 패키지의 보고 수치는 v2 기준이며, v3 는 원인 검증용으로만 사용했다.

4. 정정: Tsinghua에 머리 박스가 있다
Tsinghua Dogs 공식 페이지는 몸 전체와 머리 두 가지 바운딩 박스를 제공한다고 명시한다. 현재 인덱스는 bodybndbox만 읽고 있다.

영향은 두 가지다. SIFT 개선에서 문헌이 반복 권고하는 자세 정규화(머리 크롭)를 바로 쓸 수 있고, D의 RQ2 머리 조건도 Tsinghua 쪽에서는 실제 박스로 만들 수 있다.

확인 방법:

import glob, xml.etree.ElementTree as ET
f = sorted(glob.glob("/content/ann/**/*.xml", recursive=True))[0]
r = ET.parse(f).getroot()
for o in r.iter("object"):
    print("name:", o.findtext("name"), "| 자식 태그:", [c.tag for c in o])
5. 개선 방향 (문헌 근거 순)
방법	근거	예상 효과
색 SIFT (opponent/RGB-SIFT)	같은 BoW 틀에서 21.6% 대 회색조	가장 큼
머리 박스 크롭 (자세 정규화)	Khosla 및 ImageNet 엔트리 모두 권고	큼
공간 피라미드 (1×1 + 2×2)	Devil: IFV+SP가 FK 대비 +3 AP	중간
코드북 1024~4096	세밀 분류 표준	중간
Fisher Vector / VLAD 인코딩	Devil에서 IFV가 BoVW보다 일관되게 우위	중간
다만 이는 RQ1 범위를 넘는다. SIFT를 전통 특징의 하한선으로 두는 현재 설계라면, 현재 수치가 문헌과 일치한다는 점을 밝히는 것으로 충분하다. 개선을 시도한다면 색 SIFT 하나만 추가해도 보고서에 쓸 근거가 생긴다.

6. 보고서용 요약 문장
SIFT+BoVW의 순종 Top-1 9.5%는 선행 연구 범위 안에 있다. 회색조 SIFT BoW는 Columbia Dogs 133종에서 약 5%, 색을 포함한 opponent-SIFT BoW는 Stanford Dogs 120종에서 21.6%로 보고되었다. 본 연구의 설정(회색조, K=512)은 그 사이에 위치한다. 낮은 성능의 주된 원인은 BoVW가 색과 공간 배치를 버리고 질감 빈도만 남긴다는 점이며, 품종 구분은 바로 그 두 축에 의존한다. 같은 파이프라인의 딥 특징(DINOv3 87.9%)은 Tsinghua 공식 벤치마크 대역(77.7~86.4%)에 들어가므로, 이 격차는 구현 문제가 아니라 표현의 차이로 해석한다.

한편 믹스견 평가에서 SIFT가 찍기 기준선 아래로 내려간 것은 표현력으로 설명되지 않는다. 학습 특징과 평가 특징이 서로 다른 코드북 공간에 있었던 구현 문제이며, 해당 결과는 재추출 후 다시 산출한다.

참고 문헌
Khosla et al. (2011). Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs. https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf
Fine-grained Categorization — Short Summary of our Entry for the ImageNet Challenge 2012. https://ar5iv.arxiv.org/html/1310.4759
Chatfield et al. Return of the Devil in the Details: Delving Deep into Convolutional Nets. https://ar5iv.labs.arxiv.org/html/1405.3531
Tsinghua Dogs Dataset 공식 페이지. https://cg.cs.tsinghua.edu.cn/ThuDogs/
CS231A 프로젝트 (2016), Columbia Dogs 133종 SIFT BoW. https://web.stanford.edu/class/cs231a/prev_projects_2016/output%20(1).pdf