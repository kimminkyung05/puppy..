"""CVPR Word 템플릿(cvpr_author_kit/word/report.docx)에 paper_{ko,en}.md 를 채워 docx 를 만든다.

템플릿의 스타일 · 쪽 설정(2단, Letter) · 제목 상자 · 자동 번호(1., 1.1., [1])를 그대로 쓰고 본문만 바꾼다.
수식은 pandoc 으로 LaTeX → OMML 변환해 템플릿과 같은 eqArr "#(n)" 형식으로 번호를 단다.
사용 (report/cvpr 안에서): python build_cvpr_docx.py en   또는   python build_cvpr_docx.py ko
원고 문법: paper_ko.md 머리 주석 참고 (TITLE/AUTHORS/IDS/AFFIL, ABSTRACT, # / ##, ::: table, ::: figure, $$ $$ {#eq:..},
          [@키; @키] 인용, [[fig:..]] [[tab:..]] [[eq:..]] 참조, **굵게**, *기울임*, $수식$)
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parents[1] / "cvpr_author_kit" / "word" / "report.docx"
COL_TW = 4677                     # 칼럼 폭 (twips) = (12240 - 1440 - 902 - 544) / 2
EMU_IN = 914400

# ---------------------------------------------------------------- 참고문헌 (CVPR ieee_fullname 형식, 저자 성 알파벳순)
# 서지는 report/refs_ieee.json · computervision_filled.docx 와 같은 내용. 저자 이름은 확인된 머리글자만 쓴다.
REFS = {
    "Carion2025": "N. Carion et al. SAM 3: Segment anything with concepts. In *ICLR*, 2026.",
    "Csurka2004": "G. Csurka, C. R. Dance, L. Fan, J. Willamowski, and C. Bray. Visual categorization with bags of keypoints. In *ECCV International Workshop on Statistical Learning in Computer Vision*, 2004.",
    "Dosovitskiy2021": "A. Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2021.",
    "E2026": "G.-X. E and G.-D. Wang. Image feature embedding with a deep learning framework improves genome-wide association studies on dog endophenotypes. *Science Advances*, 12(26):eaee1088, 2026.",
    "Gao2017": "B.-B. Gao, C. Xing, C.-W. Xie, J. Wu, and X. Geng. Deep label distribution learning with label ambiguity. *IEEE TIP*, 26(6):2825–2838, 2017.",
    "Geifman2017": "Y. Geifman and R. El-Yaniv. Selective classification for deep neural networks. In *NeurIPS*, 2017.",
    "Geirhos2019": "R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In *ICLR*, 2019.",
    "Geng2016": "X. Geng. Label distribution learning. *IEEE TKDE*, 28(7):1734–1748, 2016.",
    "Goring2013": "C. Göring, A. Freytag, E. Rodner, and J. Denzler. Fine-grained categorization – short summary of our entry for the ImageNet Challenge 2012. *arXiv preprint arXiv:1310.4759*, 2013.",
    "Gunter2018": "L. M. Gunter, R. T. Barber, and C. D. L. Wynne. A canine identity crisis: Genetic breed heritage testing of shelter dogs. *PLOS ONE*, 13(8):e0202633, 2018.",
    "Guo2017": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In *ICML*, 2017.",
    "Heinz2001": "D. C. Heinz and C.-I. Chang. Fully constrained least squares linear spectral mixture analysis method for material quantification in hyperspectral imagery. *IEEE TGRS*, 39(3):529–545, 2001.",
    "Khosla2011": "A. Khosla, N. Jayadevaprakash, B. Yao, and L. Fei-Fei. Novel dataset for fine-grained image categorization: Stanford Dogs. In *CVPR Workshop on Fine-Grained Visual Categorization*, 2011.",
    "Lippert2017": "C. Lippert et al. Identification of individuals by trait prediction using whole-genome sequencing data. *PNAS*, 114(38):10166–10171, 2017.",
    "Lowe2004": "D. G. Lowe. Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2):91–110, 2004.",
    "Ma2026": "J. Ma et al. Bridging morphology and genomics: A rapid image-based assessment of genomic admixture in the endangered gayal (Bos frontalis). *bioRxiv*, 2026.",
    "Morrill2022": "K. Morrill et al. Ancestry-inclusive dog genomics challenges popular breed stereotypes. *Science*, 376(6592):eabk0639, 2022.",
    "Olson2015": "K. R. Olson et al. Inconsistent identification of pit bull-type dogs by shelter staff. *The Veterinary Journal*, 206(2):197–202, 2015.",
    "Oquab2024": "M. Oquab et al. DINOv2: Learning robust visual features without supervision. *TMLR*, 2024.",
    "Radford2021": "A. Radford et al. Learning transferable visual models from natural language supervision. In *ICML*, 2021.",
    "Rando2024": "H. M. Rando, K. Graim, G. Hampikian, and C. S. Greene. Many direct-to-consumer canine genetic tests can identify the breed of purebred dogs. *JAVMA*, 262(5):1–8, 2024.",
    "Simeoni2025": "O. Siméoni et al. DINOv3. *arXiv preprint arXiv:2508.10104*, 2025.",
    "Trager2023": "M. Trager, P. Perera, L. Zancato, A. Achille, P. Bhatia, and S. Soatto. Linear spaces of meanings: Compositional structures in vision-language models. In *ICCV*, pages 15395–15404, 2023.",
    "UF2026": "University of Florida Shelter Medicine Program. Dog breed identification photo chart: DNA results and photos. https://sheltermedicine.vetmed.ufl.edu/research/current-studies/dog-breed-identification/dog-breed-dna-and-survey-results/. Accessed Nov. 22, 2026.",
    "Verma2019": "V. Verma et al. Manifold Mixup: Better representations by interpolating hidden states. In *ICML*, 2019.",
    "Voith2009": "V. L. Voith, E. Ingram, K. Mitsouras, and K. Irizarry. Comparison of adoption agency breed identification and DNA breed identification of dogs. *Journal of Applied Animal Welfare Science*, 12(3):253–262, 2009.",
    "Voith2013": "V. L. Voith, R. Trevejo, S. Dowling-Guyer, C. Chadik, A. Marder, V. Johnson, and K. Irizarry. Comparison of visual and DNA breed identification of dogs and inter-observer reliability. *American Journal of Sociological Research*, 3(2):17–29, 2013.",
    "Wang2020": "H. Wang, X. Wu, Z. Huang, and E. P. Xing. High-frequency component helps explain the generalization of convolutional neural networks. In *CVPR*, 2020.",
    "Wilson1927": "E. B. Wilson. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158):209–212, 1927.",
    "Yin2019": "D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer. A Fourier perspective on model robustness in computer vision. In *NeurIPS*, 2019.",
    "Zhang2018": "H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz. mixup: Beyond empirical risk minimization. In *ICLR*, 2018.",
    "Zou2020": "D.-N. Zou, S.-H. Zhang, T.-J. Mu, and M. Zhang. A new dataset of dog breed images and a benchmark for fine-grained classification. *Computational Visual Media*, 6(4):477–487, 2020.",
}
SORT_NAME = {"Simeoni2025": "Simeoni", "Goring2013": "Goring"}

LABELS = {
    "en": dict(fig="Figure", tab="Table", eq="", refs="References", abstract="Abstract"),
    "ko": dict(fig="그림", tab="표", eq="식 ", refs="References", abstract="Abstract"),
}
HILITE = re.compile(r"\[(?:TBD|확인 필요|Check)[^\]]*\]")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def smart_quotes(s):
    out, open_ = [], True
    for ch in s:
        if ch == '"':
            out.append("“" if open_ else "”"); open_ = not open_
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------- 원고 읽기
def parse(md):
    meta, blocks = {}, []
    lines = md.splitlines()
    i = 0
    para = []

    def flush():
        if para:
            blocks.append(("p", " ".join(para))); para.clear()

    while i < len(lines):
        ln = lines[i].rstrip()
        m = re.match(r"^(TITLE|AUTHORS|IDS|AFFIL):\s*(.*)$", ln)
        if m and not blocks:
            meta[m.group(1)] = m.group(2); i += 1; continue
        if ln == "ABSTRACT":
            flush(); blocks.append(("abstract_head", "")); i += 1; continue
        if not ln.strip():
            flush(); i += 1; continue
        if ln.startswith("## "):
            flush(); blocks.append(("h2", ln[3:].strip())); i += 1; continue
        if ln.startswith("# "):
            flush(); blocks.append(("h1", ln[2:].strip())); i += 1; continue
        m = re.match(r"^\$\$(.+)\$\$\s*\{#(eq:[\w-]+)\}\s*$", ln)
        if m:
            flush(); blocks.append(("eq", m.group(1).strip(), m.group(2))); i += 1; continue
        m = re.match(r"^::: (table|figure) \{#([\w:-]+)\}(.*)$", ln)
        if m:
            flush()
            kind, label, opts = m.group(1), m.group(2), dict(re.findall(r"(\w+)=(\S+)", m.group(3)))
            body = []
            i += 1
            while lines[i].rstrip() != ":::":
                body.append(lines[i].rstrip()); i += 1
            i += 1
            if kind == "figure":
                blocks.append(("figure", label, opts, " ".join(body)))
            else:
                cap = [b for b in body if not b.startswith("|")]
                rows = [b for b in body if b.startswith("|")]
                blocks.append(("table", label, opts, " ".join(cap), rows))
            continue
        para.append(ln.strip()); i += 1
    flush()
    return meta, blocks


# ---------------------------------------------------------------- 수식 (pandoc)
def omml_all(snips):
    """snips: [(latex, display)] → 각 수식의 m:oMath 안쪽 XML."""
    if not snips:
        return []
    tmp = Path(tempfile.mkdtemp())
    md = "\n\n".join((f"$${s}$$" if disp else f"${s}$") for s, disp in snips)
    (tmp / "m.md").write_text(md, encoding="utf-8")
    subprocess.run(["pandoc", str(tmp / "m.md"), "-f", "markdown", "-o", str(tmp / "m.docx")], check=True)
    x = zipfile.ZipFile(tmp / "m.docx").read("word/document.xml").decode("utf-8")
    body = x[x.index("<w:body>"):]
    paras = re.findall(r"<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>", body, re.S)
    out = []
    for p in paras:
        m = re.search(r"<m:oMath>(.*)</m:oMath>", p, re.S)
        if m:
            out.append(m.group(1))
    shutil.rmtree(tmp, ignore_errors=True)
    assert len(out) == len(snips), (len(out), len(snips))
    return out


# ---------------------------------------------------------------- 문서 조립
class Doc:
    def __init__(self, lang, meta, blocks):
        self.lang, self.meta, self.blocks = lang, meta, blocks
        self.L = LABELS[lang]
        self.num = {}
        self.cites = []
        self.media = []          # (rid, src_path, name)
        n = {"fig": 0, "tab": 0, "eq": 0}
        for b in blocks:
            if b[0] in ("figure", "table", "eq"):
                lab = b[1] if b[0] != "eq" else b[2]
                k = lab.split(":")[0]
                n[k] += 1
                self.num[lab] = n[k]
        # 인용 번호: 실제로 인용된 문헌만, 알파벳순
        text = "\n".join(str(b) for b in blocks)
        keys = set(re.findall(r"@(\w+)", " ".join(re.findall(r"\[@[^\]]+\]", text))))
        missing = keys - REFS.keys()
        assert not missing, f"REFS 에 없는 인용: {missing}"
        self.ref_order = sorted(keys, key=lambda k: (SORT_NAME.get(k, re.sub(r"\d+$", "", k)), k))
        self.ref_no = {k: i + 1 for i, k in enumerate(self.ref_order)}
        # 수식 모으기
        self.math = []
        for b in blocks:
            if b[0] == "eq":
                self.math.append((b[1], True))
            for s in [x for x in b[1:] if isinstance(x, str)] + ([r for r in b[4]] if b[0] == "table" else []):
                for mm in re.findall(r"(?<!\$)\$([^$]+)\$(?!\$)", s if b[0] != "eq" else ""):
                    self.math.append((mm, False))
        conv = omml_all(self.math)
        self.math_xml = {}
        for (s, d), xml in zip(self.math, conv):
            self.math_xml[(s, d)] = xml

    # -------- 글자 단위
    def cite(self, inner):
        nums = sorted({self.ref_no[k] for k in re.findall(r"@(\w+)", inner)})
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(map(str, nums[i:j + 1])))
            i = j + 1
        return "[" + ", ".join(parts) + "]"

    def ref(self, lab):
        k = lab.split(":")[0]
        n = self.num[lab]
        if k == "eq":
            return f"{self.L['eq']}({n})"
        return f"{self.L[k]} {n}"

    def runs(self, text, base=None, size=None):
        """인라인 문법 → w:r / m:oMath 목록."""
        base = dict(base or {})
        text = re.sub(r"\[\[([\w:-]+)\]\]", lambda m: self.ref(m.group(1)), text)
        text = re.sub(r"\[(@[^\]]+)\]", lambda m: self.cite(m.group(1)), text)
        text = smart_quotes(text)
        out = []
        tokens = re.split(r"(\$[^$]+\$|\*\*.+?\*\*|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))", text)
        for tok in tokens:
            if not tok:
                continue
            if tok.startswith("$") and tok.endswith("$") and len(tok) > 1:
                out.append(f"<m:oMath>{self.math_xml[(tok[1:-1], False)]}</m:oMath>")
            elif tok.startswith("**"):
                out += self.plain(tok[2:-2], dict(base, b=True), size)
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                out += self.plain(tok[1:-1], dict(base, i=True), size)
            else:
                out += self.plain(tok, base, size)
        return "".join(out)

    def plain(self, s, fmt, size):
        out = []
        for piece in re.split(f"({HILITE.pattern}|<br>)", s):
            if not piece:
                continue
            if piece == "<br>":
                out.append("<w:r><w:br/></w:r>"); continue
            rpr = ""
            if fmt.get("b"):
                rpr += "<w:b/><w:bCs/>"
            if fmt.get("i"):
                rpr += "<w:i/><w:iCs/>"
            if HILITE.fullmatch(piece):
                rpr += '<w:highlight w:val="yellow"/>'
            if size:
                rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
            out.append(f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}'
                       f'<w:t xml:space="preserve">{esc(piece)}</w:t></w:r>')
        return out

    # -------- 블록
    def p_text(self, text):
        return f'<w:p><w:pPr><w:pStyle w:val="Text"/></w:pPr>{self.runs(text)}</w:p>'

    def p_h(self, text, level):
        sz = 24 if level == 1 else 22
        style = "Heading1" if level == 1 else "Heading2"
        rpr = f'<w:b/><w:bCs/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
        nonum = '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>' if text == self.L["refs"] else ""
        return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{nonum}<w:rPr>{rpr}</w:rPr></w:pPr>'
                f'<w:r><w:rPr>{rpr}</w:rPr><w:t>{esc(text)}</w:t></w:r></w:p>')

    def p_eq(self, latex, lab):
        inner = self.math_xml[(latex, True)]
        n = self.num[lab]
        return ('<w:p><w:pPr><w:pStyle w:val="Text"/><w:spacing w:before="80" w:after="80"/><w:ind w:firstLine="0"/></w:pPr>'
                '<m:oMathPara><m:oMath><m:eqArr><m:eqArrPr><m:maxDist m:val="1"/></m:eqArrPr><m:e>'
                f'{inner}<m:r><m:t>#</m:t></m:r><m:d><m:e><m:r><m:t>{n}</m:t></m:r></m:e></m:d>'
                '</m:e></m:eqArr></m:oMath></m:oMathPara></w:p>')

    def caption(self, kind, lab, text, keep_next):
        label = f"{self.L[kind]} {self.num[lab]}: "
        kn = "<w:keepNext/><w:keepLines/>" if keep_next else "<w:keepLines/>"   # 캡션이 단·쪽 사이에서 쪼개지지 않게
        sp ='<w:spacing w:before="60" w:after="60"/>' if kind == "tab" else '<w:spacing w:before="40" w:after="160"/>'
        return (f'<w:p><w:pPr>{kn}{sp}<w:jc w:val="both"/><w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>'
                f'{self.plain(label, {}, 18)[0]}{self.runs(text, size=18)}</w:p>')

    def figure(self, lab, opts, cap):
        src = HERE / opts["src"]
        w_in = float(opts.get("width", 3.25))
        with Image.open(src) as im:
            h_in = w_in * im.size[1] / im.size[0]
        rid = f"rIdFig{len(self.media) + 1}"
        name = f"cvpr_{self.lang}_{src.name}"
        self.media.append((rid, src, name))
        cx, cy = int(w_in * EMU_IN), int(h_in * EMU_IN)
        did = 100 + len(self.media)
        drawing = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                   f'<wp:docPr id="{did}" name="Figure {did}"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                   '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                   '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                   f'<pic:nvPicPr><pic:cNvPr id="{did}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
                   f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                   f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
                   '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>')
        pic = (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/></w:pPr>'
               f'<w:r>{drawing}</w:r></w:p>')
        return pic + self.caption("fig", lab, cap, False)

    def table(self, lab, opts, cap, rows):
        widths = [float(x) for x in opts["cols"].split(",")]
        tw = [int(COL_TW * w / sum(widths)) for w in widths]
        size = int(float(opts.get("size", 8)) * 2)
        body, rule_after = [], set()
        for r in rows:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells if c) and r.startswith("|-"):
                rule_after.add(len(body) - 1); continue
            body.append(cells)
        tbl = [f'<w:tbl><w:tblPr><w:tblW w:w="{sum(tw)}" w:type="dxa"/><w:jc w:val="center"/>'
               '<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/></w:tblBorders>'
               '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="30" w:type="dxa"/><w:right w:w="30" w:type="dxa"/></w:tblCellMar>'
               '<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>' + "".join(f'<w:gridCol w:w="{w}"/>' for w in tw) + '</w:tblGrid>']
        for ri, cells in enumerate(body):
            cells = cells + [""] * (len(tw) - len(cells))
            last = ri == len(body) - 1
            tr = ['<w:tr><w:trPr><w:cantSplit/></w:trPr>']
            for ci, (c, w) in enumerate(zip(cells, tw)):
                bdr = '<w:tcBorders><w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>' \
                    if ri in rule_after else ""
                jc = "left" if ci == 0 else "center"
                runs = self.runs(f"**{c}**" if ri == 0 and c else c, size=size)
                kn = "" if last else "<w:keepNext/>"
                tr.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{bdr}<w:vAlign w:val="center"/></w:tcPr>'
                          f'<w:p><w:pPr>{kn}<w:wordWrap w:val="1"/><w:spacing w:before="10" w:after="10"/><w:jc w:val="{jc}"/>'
                          f'<w:rPr><w:sz w:val="{size}"/></w:rPr></w:pPr>{runs}</w:p></w:tc>')
            tr.append("</w:tr>")
            tbl.append("".join(tr))
        tbl.append("</w:tbl>")
        after = '<w:p><w:pPr><w:spacing w:before="0" w:after="60"/><w:rPr><w:sz w:val="8"/></w:rPr></w:pPr></w:p>'
        return self.caption("tab", lab, cap, True) + "".join(tbl) + after

    def title_box(self, xml_tpl):
        m = self.meta
        names = m.get("AUTHORS", "").split("|")
        ids = m.get("IDS", "").split("|")
        n = len(names)
        box_tw = 9893
        tabs = "".join(f'<w:tab w:val="center" w:pos="{int(box_tw * (2 * k + 1) / (2 * n))}"/>' for k in range(n))

        def line(items, sz=24, mono=False):
            r = ""
            for it in items:
                f = '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>' if mono else ""
                r += (f'<w:r><w:rPr>{f}<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr><w:tab/>'
                      f'<w:t xml:space="preserve">{esc(it.strip())}</w:t></w:r>')
            return (f'<w:p><w:pPr><w:pStyle w:val="Authors"/><w:tabs>{tabs}</w:tabs><w:spacing w:after="0"/>'
                    f'<w:jc w:val="left"/></w:pPr>{r}</w:p>')

        content = ('<w:p><w:pPr><w:pStyle w:val="Heading"/></w:pPr></w:p>'
                   '<w:p><w:pPr><w:pStyle w:val="Heading"/></w:pPr>'
                   f'<w:r><w:rPr><w:b/><w:bCs/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr><w:t>{esc(m["TITLE"])}</w:t></w:r></w:p>'
                   '<w:p/><w:p/>' + line(names) + (line(ids, 20, True) if m.get("IDS") else "") +
                   f'<w:p><w:pPr><w:pStyle w:val="Authors"/><w:spacing w:after="0"/></w:pPr>'
                   f'<w:r><w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr><w:t>{esc(m.get("AFFIL", ""))}</w:t></w:r></w:p><w:p/>')
        return re.sub(r"<w:txbxContent>.*?</w:txbxContent>", lambda _: f"<w:txbxContent>{content}</w:txbxContent>",
                      xml_tpl, flags=re.S)

    def body(self):
        out = []
        it = iter(self.blocks)
        for b in it:
            k = b[0]
            if k == "abstract_head":
                out.append(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="24"/></w:rPr>'
                           f'<w:t>{self.L["abstract"]}</w:t></w:r></w:p>'
                           '<w:p><w:pPr><w:pStyle w:val="Abstract"/><w:ind w:firstLine="0"/></w:pPr></w:p>')
                b = next(it)
                ital = {"i": True} if self.lang == "en" else {}
                out.append(f'<w:p><w:pPr><w:ind w:firstLine="202"/><w:jc w:val="both"/></w:pPr>{self.runs(b[1], ital)}</w:p>'
                           '<w:p><w:pPr><w:ind w:firstLine="202"/><w:jc w:val="both"/></w:pPr></w:p>')
            elif k == "h1":
                out.append(self.p_h(b[1], 1))
            elif k == "h2":
                out.append(self.p_h(b[1], 2))
            elif k == "p":
                out.append(self.p_text(b[1]))
            elif k == "eq":
                out.append(self.p_eq(b[1], b[2]))
            elif k == "figure":
                out.append(self.figure(b[1], b[2], b[3]))
            elif k == "table":
                out.append(self.table(b[1], b[2], b[3], b[4]))
        out.append(self.p_h(self.L["refs"], 1))
        for key in self.ref_order:
            out.append(f'<w:p><w:pPr><w:pStyle w:val="References"/></w:pPr>{self.runs(REFS[key], size=18)}</w:p>')
        return "".join(out)


def build(lang):
    meta, blocks = parse((HERE / f"paper_{lang}.md").read_text(encoding="utf-8"))
    doc = Doc(lang, meta, blocks)
    z = zipfile.ZipFile(TEMPLATE)
    files = {n: z.read(n) for n in z.namelist()}
    x = files["word/document.xml"].decode("utf-8")
    head = x[:x.index("<w:body>") + len("<w:body>")]
    rest = x[x.index("<w:body>") + len("<w:body>"):]
    first_p = re.match(r"<w:p\b.*?</mc:AlternateContent>.*?</w:p>", rest, re.S).group(0)
    sect = re.search(r"<w:sectPr\b.*?</w:sectPr>", rest, re.S).group(0)
    new = head + doc.title_box(first_p) + doc.body() + sect + "</w:body></w:document>"
    files["word/document.xml"] = new.encode("utf-8")
    # 동아시아 글꼴: 바탕 (한글 본문, 저자 이름)
    st = files["word/styles.xml"].decode("utf-8")
    st = st.replace('w:eastAsia="DengXian"', 'w:eastAsia="Batang"').replace(
        '<w:lang w:val="en-DE" w:eastAsia="en-US"', '<w:lang w:val="en-US" w:eastAsia="ko-KR"')
    if lang == "ko":   # 한글은 글자 단위로 줄을 나눠 양쪽 정렬의 빈칸을 줄인다
        st = st.replace('<w:pPr><w:suppressAutoHyphens/><w:autoSpaceDE w:val="0"/></w:pPr>',
                        '<w:pPr><w:suppressAutoHyphens/><w:wordWrap w:val="0"/><w:autoSpaceDE w:val="0"/></w:pPr>', 1)
    files["word/styles.xml"] = st.encode("utf-8")
    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    for rid, src, name in doc.media:
        files[f"word/media/{name}"] = src.read_bytes()
        rels = rels.replace("</Relationships>",
                            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                            f'Target="media/{name}"/></Relationships>')
    files["word/_rels/document.xml.rels"] = rels.encode("utf-8")
    ct = files["[Content_Types].xml"].decode("utf-8")
    if 'Extension="png"' not in ct:
        ct = ct.replace("<Types ", "<Types ", 1).replace(
            "<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
    files["[Content_Types].xml"] = ct.encode("utf-8")
    core = files["docProps/core.xml"].decode("utf-8")
    core = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{esc(meta['TITLE'])}</dc:title>", core)
    core = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator></dc:creator>", core)
    core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", core)
    files["docProps/core.xml"] = core.encode("utf-8")
    out = HERE / f"paper_{lang}.docx"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
        for n, data in files.items():
            if n.endswith("/") or n.endswith(".DS_Store"):
                continue
            zo.writestr(n, data)
    print("저장:", out, f"(그림 {len(doc.media)}, 표 {sum(1 for b in blocks if b[0] == 'table')}, "
                        f"수식 {sum(1 for b in blocks if b[0] == 'eq')}, 참고문헌 {len(doc.ref_order)})")


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["en", "ko"]:
        build(lang)
