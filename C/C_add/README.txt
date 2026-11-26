C파트 학습 모델 (12개 = 추출기 4종 x 시드 3)

RQ1-{sift,clip,dinov2,dinov3}-mix-v2-s{1,2,3}-lr3e-3.pt

키: state(가중치), classes(127종 순서), mu/sd(표준화 통계),
    seed, lr, epochs, kmax, alpha

사용:
  ck = torch.load(path, map_location="cpu", weights_only=False)
  model = nn.Linear(len(ck["mu"]), len(ck["classes"]))
  model.load_state_dict(ck["state"]); model.eval()
  z = (X - ck["mu"]) / ck["sd"]
  P = softmax(model(tensor(z)))   # P[:, i] -> ck["classes"][i]

검증: 보고 수치를 산출한 모델과 동일하다. dev 예측 확률 최대 차이 0.000000,
      1위 예측 일치율 1.0000 (9개 조합 전부).
      재현에는 GPU 필요 (검증 환경: torch 2.14.1+cu130, Tesla T4).
