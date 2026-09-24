# 3단계 — PaddleOCR-VL 실행

**상태**: 설정 확정 / 본 실행 대기
**선행**: 2단계 완료 (`granite_raw.jsonl` 150건, 실패 0)
**실행 위치**: Kaggle Notebook (Granite와 **별도 커널**)
**실행일**: 2026-09-23

---

## 1. 목적

PaddleOCR-VL-1.6으로 **같은 150장, 같은 순서**를 파싱해 원본 출력을 저장한다.
`run_list.csv` 상위 150건을 그대로 사용한다.

---

## 2. 모델 비교

| | Granite Docling | PaddleOCR-VL-1.6 |
|---|---|---|
| 크기 | 258M | **0.9B (3.5배)** |
| 다운로드 | 515MB | **1.92GB** |
| 개발사 | IBM (미국) | 바이두 (중국) |
| 기반 | Granite 자체 | ERNIE-4.5 |
| 라이선스 | Apache 2.0 | Apache 2.0 |
| 벤치마크 | 상위권 아님 | OmniDocBench v1.6 1위 96.33% |
| 장당 소요 | 26.3초 | **18.0초** |

**모델이 3.5배 큰데도 더 빠르다.** 출력 토큰 수가 적기 때문이다.

---

## 3. 브리핑 8절 정정 — 페이지 단위 파싱 가능 ★

브리핑에 이렇게 적혀 있었다.

> PaddleOCR-VL-1.6은 transformers만으로 실행 가능하나
> **요소 단위 인식(ocr/table/chart/formula/spotting/seal)만 지원**되고
> 페이지 단위 파싱은 안 됨 → README에 한 줄 명시

**실측 결과 사실과 다르다.**

`Table Recognition:` 한 번 호출로 페이지 전체가 나온다.
헤더, 판매자, 고객, ITEMS 표, SUMMARY 표까지 1,351자에 모두 담긴다.

### 영향

| | 브리핑 전제 | 실제 |
|---|---|---|
| 호출 횟수 | 장당 2회 이상 | **1회** |
| 속도 불리 | 예상됨 | **오히려 Granite보다 빠름** |

브리핑 9절 축 2(배포 부담)의 근거 중 **호출 횟수 항목은 철회한다.**

남은 근거는 유효하다.

```
파싱만 필요       → PaddleOCR-VL 0.9B 하나
필드 추출까지     → PP-ChatOCRv4 (PP-Structure + 벡터DB + ERNIE-4.5-300B)
Granite           → Docling 258M으로 파싱을 덮음
모델 크기         → 258M vs 0.9B (3.5배)
```

**이 정정은 PaddleOCR에 유리한 방향이다. 그대로 기재한다 (브리핑 13절).**

---

## 4. 로딩 — `trust_remote_code` 문제

### 발생한 오류

```
AttributeError: 'PaddleOCRVLConfig' object has no attribute 'text_config'
```

### 원인

transformers 5.0.0에 **PaddleOCR-VL이 이미 내장**되어 있다.
`trust_remote_code=True`를 주면 설정은 HF 원격 코드로,
모델은 transformers 내장 코드로 만들어져 서로 어긋난다.

```
원격 config   → text_config 없음
내장 modeling → text_config 참조   → AttributeError
```

### 해결

`trust_remote_code`를 **프로세서와 모델 양쪽에서 제거**한다.

```python
proc  = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("cuda").eval()
```

제거 후 `text_config 존재: True`로 정상 로드된다.

> `rope_parameters` 관련 경고(`mrope_section` 미인식)는 출력되나 동작에 영향 없다.

---

## 5. 프롬프트 탐색 ★

2단계에서 Granite 프롬프트를 5개 시험했으므로, **동일한 수준의 탐색**을 수행한다.
브리핑 5절의 공정성 요건이다.

### 5.1 5개 비교 (동일 이미지 `batch1-0182`)

| 프롬프트 | 소요 | 길이 | 형식 | 판정 |
|---|---|---|---|---|
| `OCR:` | 15.1초 | 704자 | 평문 | 고객 정보 누락 |
| **`Table Recognition:`** | **18.0초** | **1,351자** | **OTSL 태그** | **채택** |
| `Text Recognition:` | 15.8초 | 766자 | 평문 | 최종 후보 |
| `OCR this image.` | 15.7초 | 775자 | 평문 | 주소 블록 뒤섞임 |
| `Convert this document to markdown.` | 17.6초 | 927자 | 표 | **환각 발생** |

**다섯 개 모두 총액·날짜를 읽는 데는 성공했다.**
Granite가 프롬프트에 따라 표 셀 0개와 40개로 갈린 것과 대조된다.

### 5.2 탈락 사유

**`Convert this document to markdown.`** — 출력에 `Rating`이 반복 등장한다.
송장에 없는 단어이며 **환각**이다. 제외.

**`OCR:`** — 고객 블록(`Rodriguez-Morgan`)이 통째로 빠진다.

**`OCR this image.`** — 판매자와 고객 주소가 뒤섞여 출력된다.

### 5.3 최종 2개 대조

| | `Table Recognition:` | `Text Recognition:` |
|---|---|---|
| 총액 `127,31` | 있음 | 있음 |
| 날짜 `10/02/2020` | 있음 | 있음 |
| 상호 `Campbell...` | 있음 | 있음 |
| 고객 `Rodriguez-Morgan` | 있음 | 있음 |
| **출력 형식** | **OTSL 태그** | 평문 |
| 소요 | 18.0초 | 15.8초 |

**값은 둘 다 맞는다. 형식에서 갈린다.**

### 5.4 `Table Recognition:` 채택 근거

```
Granite     <otsl>...<fcel>115,74<fcel>11,57<fcel>127,31<nl></otsl>
PaddleOCR   <fcel>10%<lcel><fcel>115,74<fcel>11,57<fcel>127,31<lcel><nl>
             ↑ 동일한 태그 체계
```

브리핑 5절이 요구하는 것은 **"두 모델 출력에 동일하게 적용되는 추출기 하나"** 다.

`Text Recognition:`을 택하면 한쪽은 태그, 한쪽은 평문이 되어
추출 경로가 갈라진다. 그러면 **"어느 쪽 파서를 더 잘 짰나"가 결과에 섞인다.**
브리핑이 3층에 Granite Vision을 두지 말라고 한 것과 같은 문제다.

**두 모델이 같은 태그를 뱉는 쪽을 택해 추출기를 하나로 유지한다.**
2.2초 차이는 그 대가로 감당한다.

---

## 6. 확정 설정

| 항목 | 값 |
|---|---|
| 모델 | `PaddlePaddle/PaddleOCR-VL-1.6` |
| 클래스 | `AutoModelForImageTextToText` |
| `trust_remote_code` | **사용 안 함** |
| dtype | `bfloat16` |
| **프롬프트** | **`Table Recognition:`** |
| `max_new_tokens` | 3000 |
| `skip_special_tokens` | **True** (Granite는 False) |
| GPU | Tesla T4 |
| transformers | 5.0.0 |
| 표본 | 상위 150건 (Granite와 동일) |

### Granite와 다른 점

| | Granite | PaddleOCR |
|---|---|---|
| `attn_implementation` | `sdpa` 지정 | 기본값 |
| `skip_special_tokens` | False | **True** |

Granite는 `<doctag>` 등 특수 토큰이 구조를 담으므로 보존해야 한다.
PaddleOCR은 `<fcel>` 등이 일반 토큰이라 특수 토큰을 걸러도 남는다.

### 커널 분리

transformers 버전이 같아도 **커널은 분리한다.**
한 세션에 두 모델(515MB + 1.92GB)을 올리면 T4 16GB가 빠듯하고,
한쪽 실패가 다른 쪽에 번진다.

---

## 7. 출력 형식

`paddle_raw.jsonl` — Granite와 동일 구조.

```json
{"file_name": "batch1-0182", "seq": 1, "ok": true,
 "output": "<fcel>Invoice no: 89790497<lcel>...", "elapsed": 18.0}
```

객체가 아니라 **문자열**이다. 한 번 호출로 끝나므로 호출별 분리가 불필요하다.

---

## 8. 실행 방식

2단계의 교훈을 반영한다.

- **Save Version → Save & Run All** 로 실행
  (대화형 세션은 종료 시 `/kaggle/working/` 파일이 소멸한다. 2단계에서 53분 작업분을 잃었다)
- **Accelerator를 GPU로 설정했는지 반드시 확인**
  (2단계에서 `accelerator: none` 으로 커밋해 0초 실패)
- 실험용 셀은 남기지 않는다 (Save & Run All은 모든 셀을 재실행한다)
- 매 건 `flush()`

### 예상 소요

```
18.0초 × 150장 ≈ 45분
+ 설치·모델 다운로드·커밋 마무리 약 8분
= 약 53분
```

---

## 9. 중단 조건

- 모델 로딩 실패
- 실패율 10% 초과
- 장당 소요가 60초를 넘어 150장이 2.5시간 초과
  → 표본을 100장으로 축소하고 **Granite도 동일하게 100장으로 잘라 재채점**

---

## 10. README 반영 문구

> PaddleOCR-VL-1.6도 Granite와 동일하게 5개 프롬프트를 비교했다.
> 다섯 개 모두 총액과 날짜를 읽었으나, `"Convert this document to markdown."`에서는
> 문서에 없는 `Rating`이 반복 출력되는 환각이 발생해 제외했다.
> 최종적으로 `"Table Recognition:"`을 채택했는데, 값의 정확도가 아니라
> **출력 형식 때문**이다. 이 프롬프트는 Granite의 DocTags와 동일한 OTSL 태그를
> 사용하므로, 두 모델 출력에 같은 추출기를 적용할 수 있다.
>
> 브리핑 단계에서는 PaddleOCR-VL이 페이지 단위 파싱을 지원하지 않아
> 요소별 다중 호출이 필요할 것으로 보았으나, 실측 결과 한 번 호출로
> 페이지 전체가 출력됐다. 장당 소요도 18.0초로 Granite의 26.3초보다 빨랐다.
> 이 정정은 PaddleOCR에 유리한 방향이며 그대로 기재한다.
>
> transformers 5.0.0에 PaddleOCR-VL이 내장되어 있어 `trust_remote_code=True`를
> 주면 원격 설정과 내장 모델 코드가 충돌한다. 해당 인자를 제거해야 로드된다.

---

## 11. 다음 단계

4단계에서 두 출력을 **같은 추출기**에 통과시켜 채점한다.

```
granite_raw.jsonl  (DocTags, <otsl><fcel>...)  ──┐
                                                  ├→ 단일 추출기 → 필드값
paddle_raw.jsonl   (OTSL,    <fcel>...)        ──┘
```

양쪽 모두 `<fcel>` `<lcel>` `<nl>` 태그를 쓰므로 공통 처리가 가능하다.