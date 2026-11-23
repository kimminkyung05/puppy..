# A파트 최종 Hold 평가 결과

## 1. 실행 상태

- 합의된 `EVALUATION_POLICY.md` 기준으로 hold를 **한 번만** 채점했다.
- 평가 후 모델, 매핑, 동률 처리 또는 제외 기준을 변경하지 않았다.
- 재실행 방지 기록: `results/HOLD_EVALUATED_ONCE.json`

## 2. 평가 조건

- Hold: 80마리
- 평가 가능: 71마리
- 지배 품종이 전부 Tsinghua 범위 밖이어서 제외: 9마리
- Coverage: 71/80 = **88.75%**
- face/body 확률을 개체 단위로 평균
- seed 1·2·3 확률을 평균한 앙상블을 대표 결과로 사용
- 공동 1위 중 매핑 가능한 품종 하나를 맞히면 정답
- DNA 퍼센트는 부분 정답이므로 합을 100으로 보정하지 않음

## 3. Mixed-dog Hold Top-1

| 특징 | 정답/평가 가능 | Top-1 | Wilson 95% CI | Coverage |
|---|---:|---:|---:|---:|
| CLIP | 10/71 | **14.1%** | 7.8–24.0% | 88.75% |
| DINOv3 | 9/71 | **12.7%** | 6.8–22.4% | 88.75% |
| DINOv2 | 8/71 | **11.3%** | 5.8–20.7% | 88.75% |
| SIFT | 1/71 | **1.4%** | 0.2–7.6% | 88.75% |

### Seed별 범위

| 특징 | Seed 1 | Seed 2 | Seed 3 | Seed 앙상블 |
|---|---:|---:|---:|---:|
| CLIP | 14.1% | 15.5% | 14.1% | 14.1% |
| DINOv2 | 11.3% | 11.3% | 11.3% | 11.3% |
| DINOv3 | 11.3% | 12.7% | 11.3% | 12.7% |
| SIFT | 1.4% | 1.4% | 1.4% | 1.4% |

## 4. Purebred Validation Top-1

| 특징 | Top-1 평균 ± 표준편차 |
|---|---:|
| DINOv3 | **87.94% ± 0.09%p** |
| DINOv2 | **85.63% ± 0.05%p** |
| CLIP | **72.45% ± 0.22%p** |
| SIFT | **9.53% ± 0.15%p** |

순종견과 믹스견 결과는 서로 다른 데이터셋과 정답 구조를 사용하므로 별도 표로 보고한다.

## 5. 해석

- 모든 딥 특징은 순종견에서는 높은 정확도를 보였지만 실제 믹스견 지배 품종 Top-1에서는 성능이 크게 낮아졌다.
- 순종 성능이 가장 높은 DINOv3가 믹스견에서 가장 높지는 않았다. CLIP이 믹스견 hold에서 수치상 가장 높았지만, CLIP·DINOv2·DINOv3의 95% 신뢰구간이 크게 겹치므로 세 딥 특징의 순위를 강하게 주장하면 안 된다.
- SIFT는 순종과 믹스견 모두에서 낮아 고전 특징 하한선 역할을 한다.
- 모델이 출력할 수 없는 지배 품종 때문에 9마리가 제외되었으므로 Top-1 결과는 반드시 Coverage 88.75%와 함께 보고해야 한다.

## 6. 한계

- DNA 파일에는 상위 4개 혈통만 기록되어 있어 완전한 혈통 분포 정답이 아니다.
- 80마리 중 71마리만 Top-1 평가가 가능해 신뢰구간이 넓다.
- 같은 개의 face/body 두 장만 사용해 사진 다양성이 제한적이다.
- 품종 매핑표에 유사 매핑과 범위 밖 품종이 포함되어 있다.
- 본 결과는 외형 기반 Top-1 지배 품종 추정이며 DNA 검사를 대체하지 않는다.

## 7. 결과 파일

```text
results/hold_top1_individual_seeds.csv
results/hold_top1_seed_ensemble_summary.csv
results/final_purebred_vs_mixed_top1.csv
results/final_purebred_vs_mixed_top1.png
results/RQ1-*-seed-ensemble_hold_details.csv
results/HOLD_EVALUATED_ONCE.json
```
