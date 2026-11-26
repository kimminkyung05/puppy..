# CVPR 형식 원고 (report/cvpr)

CVPR Word 템플릿(`cvpr_author_kit/word/report.docx`, 2단 · Letter · Times 10pt)에 맞춘 8쪽 이내(참고문헌 제외) 원고.

| 파일 | 내용 |
|---|---|
| `paper_ko.md` | 한국어 원고 (기준 원고 `report/ko/`를 8쪽 분량으로 줄인 판) |
| `paper_en.md` | 영어 원고 (`paper_ko.md`와 같은 내용, `Academic English Writing.md` 문체) |
| `paper_ko.docx`, `paper_en.docx` | 빌드 결과 |
| `figures/{ko,en}/` | 그림 원본. 칼럼 폭 3.25 in, 인쇄 크기 기준 최소 7 pt. PNG(300 dpi, docx용) + PDF(벡터) |
| `make_cvpr_figures.py` | 그림 생성. `B/results/`의 집계 CSV만 읽는다 |
| `build_cvpr_docx.py` | 원고 → docx. 템플릿 스타일 · 제목 상자 · 자동 번호를 그대로 쓰고, 수식은 pandoc으로 OMML 변환 |

## 다시 만들기 (report/cvpr 안에서)

```powershell
..\..\.venv\Scripts\python.exe make_cvpr_figures.py
..\..\.venv\Scripts\python.exe build_cvpr_docx.py en ko
```

`pandoc`이 PATH에 있어야 한다. 숫자는 `report/ko/`, `computervision_filled.docx`와 같고, 바꿀 때는 두 원고를 함께 고친다.

## 원고 문법

- 머리: `TITLE:` `AUTHORS: a | b` `IDS:` `AFFIL:` 다음 `ABSTRACT` 한 줄과 초록 문단
- `# 장`, `## 절` (번호는 템플릿이 자동으로 붙인다), 절 안은 `**머리말.**`
- 수식: `$$ LaTeX $$ {#eq:이름}`, 본문 수식 `$...$`
- 표: `::: table {#tab:이름} cols=열 너비(in, 합 3.25로 맞춤) size=8` → 캡션 줄 → `| ... |` 행, `|---|`는 가로줄, 셀 안 줄바꿈 `<br>`
- 그림: `::: figure {#fig:이름} src=figures/en/x.png width=3.25` → 캡션
- 참조 `[[fig:이름]]` `[[tab:이름]]` `[[eq:이름]]`, 인용 `[@Zou2020; @Radford2021]` (키는 `build_cvpr_docx.py`의 `REFS`. 인용한 문헌만 성 알파벳순으로 번호를 매긴다)
- `[TBD: ...]`, `[확인 필요: ...]`, `[Check: ...]`는 노란 형광으로 표시된다
