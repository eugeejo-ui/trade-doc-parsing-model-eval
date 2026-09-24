"""
rules.py - 서류 대조 규칙

두 곳에서 '같은 구현'으로 쓰인다.
  0단계: BPI Challenge 2019 실데이터로 금액 대조 규칙 검증
  5단계: 케이스 1의 파싱 결과에 적용해 하자 탐지

따로 만들면 0단계의 검증이 5단계로 전이되지 않으므로,
amount_matches()는 값 개수에 무관하게 동작한다.
  BPI      : (PO, GR, INV)              3개
  케이스 1 : (송장 파싱값, LC 조건값)    2개

검증 상태
  amount_matches : 0단계에서 BPI 실데이터 8,564건으로 검증됨
  date_before    : 미검증 - BPI에 납기/마감 필드 없음
  name_matches   : 미검증 - BPI의 Vendor/Name은 케이스 단위 단일값
"""
from __future__ import annotations
import math, re, unicodedata

DEFAULT_EPS = 0.01   # 0단계 측정: 0.001~10.0 범위에서 판정 불변


def amount_matches(*values, eps: float = DEFAULT_EPS, rel: float = 0.0) -> bool:
    """금액들이 허용오차 안에서 모두 일치하면 True. 2개든 3개든 동일 동작.
    eps: 절대 허용오차 / rel: 상대 허용오차(0.05=±5%). 둘 중 큰 쪽을 적용."""
    nums = _clean(values)
    if len(nums) < 2:
        raise ValueError(f"금액 대조에는 값이 2개 이상 필요합니다 (받은 값 {len(nums)}개)")
    tol = max(eps, rel * max(abs(v) for v in nums))
    return (max(nums) - min(nums)) <= tol


def amount_gap(*values) -> float:
    """최대-최소 차이. 판정과 무관한 보고용 수치."""
    nums = _clean(values)
    return max(nums) - min(nums) if len(nums) >= 2 else 0.0


def amount_gap_rel(*values) -> float:
    """상대 차이(%). 기준은 절대값이 가장 큰 값."""
    nums = _clean(values)
    if len(nums) < 2:
        return 0.0
    base = max(abs(v) for v in nums)
    return (max(nums) - min(nums)) / base * 100 if base else 0.0


def date_before(actual, deadline) -> bool:
    """기한 내이면 True. [미검증] 케이스 1에서만 사용."""
    if actual is None or deadline is None:
        return False
    return actual <= deadline


_LEGAL = re.compile(r"\b(co|ltd|limited|inc|incorporated|corp|corporation|llc|plc|gmbh|sa|nv|bv|ag|pte|pty|kk)\b")


def name_matches(a, b, strict: bool = True) -> bool:
    """명칭 일치 여부. [미검증] 케이스 1에서만 사용.
    strict=True: 공백/대소문자만 정규화 (UCP 600 엄격일치에 가깝게)
    strict=False: 법인격 표기와 구두점까지 제거"""
    na, nb = _norm(a, strict), _norm(b, strict)
    return bool(na) and na == nb


def _norm(s, strict: bool) -> str:
    if s is None:
        return ""
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s)).strip().lower())
    if strict:
        return t
    t = _LEGAL.sub(" ", re.sub(r"[.,&\-/()]", " ", t))
    return re.sub(r"\s+", " ", t).strip()


def _clean(values) -> list:
    out = []
    for v in _flatten(values):
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            out.append(f)
    return out


def _flatten(obj):
    if isinstance(obj, (list, tuple, set)):
        for x in obj:
            yield from _flatten(x)
    else:
        yield obj
