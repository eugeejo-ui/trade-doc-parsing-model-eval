"""
extract_fields.py — 두 모델 출력에 동일하게 적용되는 단일 추출기

브리핑 5절의 공정성 장치.
모델별 추출 로직을 따로 만들면 "어느 쪽 파서를 더 잘 짰나"가
결과에 섞인다. 이 파일의 함수 하나가 양쪽 출력을 모두 처리한다.

입력 형식
  Granite   : DocTags  <doctag><otsl><ched>...<fcel>...<nl></otsl>  + <loc_NNN>
  PaddleOCR : OTSL     <fcel>...<lcel>...<nl>

공통 태그: <fcel> <ecel> <lcel> <nl>
"""

from __future__ import annotations

import re

SEP = "\x01"

_LOC    = re.compile(r"<loc_\d+>")
_STRUCT = re.compile(r"</?doctag>|</?otsl>|<\|end_of_text\|>|</?picture>")
_CELL   = re.compile(r"<fcel>|<ched>|<rhed>|<ecel>|<nl>")
_LCEL   = re.compile(r"<lcel>|<ucel>|<xcel>")
_TAGS   = re.compile(r"</?(?:text|section_header_level_\d|page_header|page_footer|"
                     r"title|caption|list_item|code|formula)>")
_DATE   = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")

# 금액은 항상 소수점 두 자리다 (115,74 / 127,31 / 1 234.56).
# 소수점 없는 정수(우편번호, 번지수, Tax Id)를 금액으로 오인하지 않기 위함.
_MONEY  = re.compile(r"-?\d[\d\s\u00a0\u202f]*[.,]\d{2}(?!\d)")

_STOP = ("tax id", "iban")


# ------------------------------------------------------------------ 전처리

def to_rows(raw: str) -> list[list[str]]:
    """모델 출력을 행 × 셀 구조로 평탄화한다. 두 형식 모두 처리."""
    if not raw:
        return []
    t = _LOC.sub("", raw)
    t = _STRUCT.sub("\n", t)
    t = _LCEL.sub("", t)
    t = re.sub(r"<nl>", "\n", t)
    t = _CELL.sub(SEP, t)
    t = _TAGS.sub(SEP, t)
    t = t.replace("\\n", " ")

    rows = []
    for line in t.split("\n"):
        cells = [c.strip() for c in line.split(SEP)]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def flat(rows: list[list[str]]) -> list[str]:
    return [c for r in rows for c in r]


# ------------------------------------------------------------------ 금액

def parse_amount(s):
    """1단계에서 확정한 규칙. 천 단위 = 공백, 소수점 = 점 또는 쉼표."""
    t = re.sub(r"[$€£₩]", "", str(s))
    t = re.sub(r"[\s\u00a0\u202f]", "", t)
    if not t:
        return None
    if "," in t and "." in t:
        t = (t.replace(".", "") if t.rfind(",") > t.rfind(".")
             else t.replace(",", "")).replace(",", ".")
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def extract_amount(raw: str):
    """SUMMARY 이후 영역에서 최대 금액(Gross worth / Total)을 취한다.

    두 모델의 출력 순서가 다르다.
      Granite : SUMMARY 표 → 고객 주소 블록
      Paddle  : 고객 주소 → SUMMARY 표
    SUMMARY 이후를 무제한으로 훑으면 Granite에서 우편번호를 금액으로 집는다.
    'Tax Id' / 'IBAN' 이 나오면 표가 끝난 것으로 보고 절단한다.
    """
    rows = to_rows(raw)
    cells = flat(rows)
    if not cells:
        return None

    idx = next((i for i, c in enumerate(cells) if "SUMMARY" in c.upper()), None)
    if idx is not None:
        scope = cells[idx:]
        end = next((j for j, c in enumerate(scope)
                    if any(k in c.lower() for k in _STOP)), None)
        if end is not None:
            scope = scope[:end]
    else:
        scope = cells

    vals = []
    for c in scope:
        if "%" in c:
            continue
        for m in _MONEY.findall(c):
            v = parse_amount(m)
            if v is not None and v > 0:
                vals.append(v)
    return max(vals) if vals else None


# ------------------------------------------------------------------ 날짜

def extract_date(raw: str):
    """'Date of issue' 인근의 MM/DD/YYYY. 실패 시 전체 첫 번째 날짜."""
    rows = to_rows(raw)
    cells = flat(rows)

    for i, c in enumerate(cells):
        if "date of issue" in c.lower():
            m = _DATE.search(c)
            if m:
                return m.group(1)
            for nxt in cells[i+1:i+6]:
                m = _DATE.search(nxt)
                if m:
                    return m.group(1)

    m = _DATE.search(" ".join(cells))
    return m.group(1) if m else None


# ------------------------------------------------------------------ 상호

_SKIP = ("tax id", "iban", "client", "items", "summary", "invoice no",
         "no.", "description", "qty", "net price", "net worth", "gross worth",
         "date of issue", "vat", "total")

# 주소 시작 토큰.
# 미국 주소는 번지수(숫자)로 시작하는 것이 보통이나,
# 군사우편(PSC/FPO/APO)·함정(USNV/USNS/USS/USCGC)·부대(Unit)·사서함(Box) 등
# 문자로 시작하는 형식이 있다. 두 모델 출력 모두에 동일하게 적용된다.
_ADDR_START = {
    "psc", "fpo", "apo", "dpo",
    "usnv", "usns", "uss", "usna", "uscgc", "usav", "usms",
    "unit", "box", "po", "p.o.", "pob",
    "apt", "apt.", "suite", "ste", "ste.", "floor", "fl",
}

_SELLER_PREFIX = re.compile(r"^\s*seller\s*[:：]?\s*", re.I)


def _cut_address(s: str) -> str:
    """이름과 주소가 붙어 있을 때 주소 시작 지점 앞에서 자른다.

    Granite는 이름·주소가 한 덩어리, Paddle은 \\n 으로 분리돼 있다.
    \\n 기준 절단은 Paddle에만 유리하므로 쓰지 않는다.
    숫자 시작과 주소 키워드 시작을 모두 경계로 본다.
    """
    out = []
    for tk in s.split():
        if re.match(r"^\d", tk):
            break
        if tk.lower().strip(",.") in _ADDR_START:
            break
        out.append(tk)
    return " ".join(out).strip(" ,;:")


def _is_skip(s: str) -> bool:
    low = s.lower().strip()
    return (not low) or any(low.startswith(k) or low == k for k in _SKIP)


def _clean(cand: str):
    """후보 문자열에서 상호만 남긴다. 실패 시 None."""
    if not cand or _is_skip(cand):
        return None
    name = _cut_address(cand)
    return name if len(name) >= 3 and not _is_skip(name) else None


def extract_seller(raw: str):
    """'Seller' 표기 이후의 첫 상호.

    출력 형태가 두 가지다.
      (a) 'Seller:' 행 → 다음 행에 이름            (Granite 다수, Paddle 일부)
      (b) 'Seller: 이름 주소...' 가 한 칸에 통째    (Paddle 일부)
    둘 다 처리한다.
    """
    rows = to_rows(raw)

    # (b) 같은 칸에 'Seller:' 와 이름이 함께 있는 경우 — 먼저 확인
    for row in rows:
        for c in row:
            if not _SELLER_PREFIX.match(c):
                continue
            rest = _SELLER_PREFIX.sub("", c).strip()
            name = _clean(rest)
            if name:
                return name

    # (a) 'Seller' 행 다음 행의 첫 셀
    for i, row in enumerate(rows):
        if not any("seller" in c.lower() for c in row):
            continue
        for nxt in rows[i+1:i+4]:
            name = _clean(nxt[0].strip())
            if name:
                return name

    # (c) 같은 행의 다음 칸
    for row in rows:
        for j, c in enumerate(row):
            if c.lower().startswith("seller") and j + 1 < len(row):
                name = _clean(row[j+1])
                if name:
                    return name
    return None


# ------------------------------------------------------------------ 진입점

def extract(raw: str) -> dict:
    """세 필드를 한 번에. 두 모델 출력 모두 이 함수를 통과한다."""
    return {"total":        extract_amount(raw),
            "invoice_date": extract_date(raw),
            "seller_name":  extract_seller(raw)}