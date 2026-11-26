# D 노트북 / run_d_experiment.py 에 붙여 넣는 코드 조각. 이 파일을 통째로 실행하지 않는다.
# 자세한 설명: B/B_TO_D/README.md, 추출기 본체: B/src/features/d_bridge.py

# ---------------------------------------------------------------------------
# [공통] B 코드 불러오기 + Hugging Face 로그인 (DINOv3 는 접근 승인 필요)
# ---------------------------------------------------------------------------
import sys
REPO_B = "/content/dog-mix-cv/B"          # 저장소를 clone 한 위치. B_TO_D 폴더만 받았으면 ".../B_TO_D/code"
if REPO_B not in sys.path:
    sys.path.insert(0, REPO_B)
from huggingface_hub import login; login()
from src.features.d_bridge import BExtractor

# ---------------------------------------------------------------------------
# [노트북 D_RQ2_Colab.ipynb] 셀 7(추출기 로드) 다음, 셀 10(전체 조건 채점) 전에 실행
# ---------------------------------------------------------------------------
USE_B = ["dinov3"]                         # 권장: ["clip", "dinov2", "dinov3"]
for name in USE_B:
    _extractor_cache[name] = BExtractor(name)
# 바꾼 추출기의 예전 캐시가 있으면 셀 10 이 그것을 읽는다. 먼저 확인한다.
for name in USE_B:
    old = sorted((FEATURE_ROOT / SPLIT / name).glob("*.npy"))
    print(name, "기존 캐시", len(old), "개" + (" → 지우거나 옮긴 뒤 실행" if old else ""))

# 특징만 만들 때 (분류기 채점 없이 캐시 .npy 17개만 생성)
# for cond in CONDITIONS:
#     extract_features_for_condition("dinov3", cond)

# ---------------------------------------------------------------------------
# [run_d_experiment.py] main() 에서 extractors 사전을 만든 직후에 추가
# ---------------------------------------------------------------------------
# extractors["dinov3"] = BExtractor("dinov3")
# (clip · dinov2 도 B 방식으로: extractors["clip"] = BExtractor("clip") 등)
# 캐시 위치: d_input/D_outputs/feature_cache/{추출기}/{조건}.npy
