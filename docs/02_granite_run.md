# 2단계 — Granite Docling 실행

**상태**: 본 실행 진행 중
**선행**: 1.5단계 완료 (정답지 9개 확정, `run_list.csv` 1,413건)
**실행 위치**: Kaggle Notebook
**실행일**: 2026-09-23

---

## 1. 목적

Granite Docling으로 송장 이미지를 파싱해 **원본 출력을 그대로 저장**한다.

**이 단계에서는 채점하지 않는다.** 필드 추출도 하지 않는다.

```
2단계  이미지 → Granite → DocTags 원문 저장     ← 여기
3단계  이미지 → PaddleOCR → 출력 저장
4단계  두 출력을 같은 추출기에 통과시켜 채점
```

### 왜 추출을 여기서 하지 않는가

브리핑 5절의 공정성 요건이다.
모델별로 추출 로직을 따로 짜면 "Granite 쪽 추출기를 더 잘 짜서 이긴" 상황이 된다.

---

## 2. 확정 설정

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | `ibm-granite/granite-docling-258M` | 515MB |
| 클래스 | **`AutoModelForImageTextToText`** | `AutoModelForVision2Seq`는 제거됨 |
| dtype | **`bfloat16`** | fp16은 출력이 깨짐 |
| attention | **`sdpa`** | 기본 eager보다 빠름 |
| `use_cache` | True | |
| **프롬프트** | **`Convert this page to OTSL.`** | **핵심. 4절 참조** |
| `max_new_tokens` | 3000 | 실제 생성 319토큰 |
| GPU | Tesla T4 | |
| transformers | **5.0.0** | |
| **표본** | **300장** | `run_list.csv` 상위 300건 |

---

## 3. 실행 중 발생한 문제와 해결

### 3.1 `AutoModelForVision2Seq` 제거됨

```
ImportError: cannot import name 'AutoModelForVision2Seq'
```

transformers 5.0.0에서 `AutoModelForImageTextToText`로 통합됐다.

> 부수 확인: PaddleOCR-VL-1.6도 같은 클래스와 5.x를 요구한다.
> 두 모델이 같은 버전에서 작동하지만, **커널은 그대로 분리한다.**
> 한 세션에 두 모델을 올리면 GPU 메모리가 빠듯하고 한쪽 실패가 다른 쪽에 번진다.

### 3.2 첫 실행이 10분 이상 소요 — 원인 오진

`max_new_tokens=4096`으로 설정해 종료 토큰을 만나지 못하면
4096개를 끝까지 생성한다. 22 tok/s 기준 약 6분이다.

초기에 추적에 나타난 `hidden_states.to(torch.float32)`를 보고
**"T4가 bfloat16을 지원하지 않아 느리다"고 진단했으나 오판이었다.**
RMSNorm은 어떤 dtype이든 항상 float32로 승격해 계산한다. 정상 동작이다.

진짜 원인은 토큰 수였다.

### 3.3 float16은 사용 불가

```
--- 32토큰 출력 ---
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

fp16에서 수치가 넘쳐 출력이 완전히 깨진다. Idefics3 계열의 알려진 문제다.
**bfloat16 또는 float32만 사용 가능하다.**

### 3.4 bfloat16 vs float32 — 출력 동일

| | 토큰 | 길이 | 표 셀 |
|---|---|---|---|
| bfloat16 | 319 | 994자 | 0 |
| float32 | 319 | 994자 | 0 |

**토큰 단위까지 완전히 동일하다.** 정밀도 문제가 아님이 확정됐다.
bfloat16을 채택했다.

### 3.5 `attn_implementation="sdpa"` 효과

```
기본(eager) : 11.2 tok/s
sdpa        : 19.2 tok/s
```

약 1.7배 개선됐다.

---

## 4. 프롬프트 탐색 ★

**이 단계에서 가장 중요한 발견이다.**

초기 프롬프트 `"Convert this page to docling."`에서 표가 빈 껍데기로 나왔다.

```
<otsl><loc_40><loc_177><loc_459><loc_257></otsl>
       ↑ 위치만 있고 내용이 없음
```

송장의 SUMMARY 표에 **총액**이 들어 있으므로, 이대로면 금액 규칙을 측정할 수 없다.

### 4.1 5개 프롬프트 비교

| 프롬프트 | 표 셀 | 소요 | 판정 |
|---|---|---|---|
| `Convert this page to docling.` | **0개** | 12.2초 | 표 누락 |
| **`Convert this page to OTSL.`** | **40개** | 20.5초 | **채택** |
| `Convert table to OTSL.` | 22개 | 11.7초 | 페이지 전체를 표 하나로 뭉갬 |
| `Convert chart to OTSL.` | 24개 | 7.2초 | 송장을 `<pie_chart>`로 오인 |
| `Extract all text and tables...` | 40개 | 17.8초 | **고객 정보 누락** |

**단어 하나 차이로 0셀과 40셀이 갈렸다.**

### 4.2 최종 후보 2개 비교

| | 표 | 판매자 | 고객 | 길이 |
|---|---|---|---|---|
| **`Convert this page to OTSL.`** | 40셀 | 있음 | **있음** | 1,714자 |
| `Extract all text and tables...` | 40셀 | 있음 | **없음** | 1,539자 |

후자는 고객 주소 블록이 통째로 빠진다.
현재 규칙은 `seller_name`만 쓰지만 **출력이 온전한 쪽**을 택했다.
`OTSL`이 Granite Docling의 고유 형식명이라 공식 용법에도 부합한다.

### 4.3 공정성 요건 ★

프롬프트를 5개 시험해 최선값을 찾았다.
**3단계 PaddleOCR에도 동일한 수준의 설정 탐색을 수행해야 한다.**
한쪽만 조정하면 브리핑 5절의 공정성이 깨진다.

README에 양쪽 탐색 과정을 모두 기재한다.

---

## 5. 출력 검증 — 1장 대조

`batch1-0182`로 확인했다.

```
<otsl>...<ched>No.<ched>Description<ched>Qty<ched>UM<ched>Net price
       <ched>Net worth<ched>VAT [%]<ched>Gross worth<nl>
  <fcel>1.<fcel>Free People Lace Dress...<fcel>5,00<fcel>each
       <fcel>12,00<fcel>60,00<fcel>10%<fcel>66,00<nl>
  ...
</otsl>
<section_header_level_1>SUMMARY</section_header_level_1>
<otsl>...<ched>VAT [%]<ched>Net worth<ched>VAT<ched>Gross worth<nl>
  <rhed>10%<fcel>115,74<fcel>11,57<fcel>127,31<fcel>Total<nl>
</otsl>
```

| 필드 | 모델 출력 | 정답지 | 판정 |
|---|---|---|---|
| 금액 | `127,31` | 127.31 | **일치** |
| 상호 | `Campbell, Chavez and Reynolds` | 동일 | **일치** |
| 날짜 | `10/02/2020` | 동일 | **일치** |

**규칙 3종이 읽어야 할 값이 모두 출력에 존재한다.**

### 부수 확인 — 쉼표 소수점

출력의 금액이 `115,74` `127,31` 형태다.
1단계에서 잡아낸 **쉼표 소수점 표기가 이미지에도 그대로 찍혀 있음**이 확인됐다.
`parse_amount()` 수정이 옳았다.

### 판매자 위치

왼쪽 블록이 `seller_name`, 오른쪽이 `client_name`이다.
`Rodriguez-Morgan`은 고객이다. 4단계 추출 시 좌우를 혼동하면 안 된다.

---

## 6. 표본 축소 — 1,413장 → 300장

### 6.1 실측 시간

```
장당 20.5초 (전처리 포함)
1,413장 = 8.0시간
```

Kaggle 세션 한도는 12시간, 주간 할당은 30시간이다.
Granite 8시간 + PaddleOCR 8시간 이상이면 작업 시간이 과도하다.

### 6.2 300장 선택

| 표본 | 소요 |
|---|---|
| 100장 | 35분 |
| **300장** | **1.7시간** |
| 1,413장 | 8.0시간 |

**`run_list.csv`가 금액 5분위로 층화·셔플되어 있어 상위 300건을 그대로 잘라 쓰면 전 금액대가 고루 포함된다.** 앞쪽 편중이 없다.

3단계 PaddleOCR도 **동일한 상위 300건, 동일한 순서**로 실행한다.

### 6.3 CPU 사용률이 높은 이유

실행 중 CPU 사용률이 GPU보다 높게 나타난다. 정상이다.

```
픽셀 텐서: [1, 13, 3, 512, 512]
```

1654×2339 이미지를 512×512 타일 13장으로 쪼개는 전처리가 CPU에서 돈다.
GPU는 그 뒤 토큰 생성 시에만 사용된다. 20.5초/장은 이를 포함한 값이다.

---

## 7. 출력 형식

`granite_raw.jsonl` — 한 줄에 한 건.

```json
{"file_name": "batch1-0182", "seq": 1, "ok": true,
 "output": "<doctag>...</doctag>", "elapsed": 20.5}
```

| 필드 | 내용 |
|---|---|
| `file_name` | 정답지 매칭 키 |
| `seq` | run_list 순서 |
| `ok` | 처리 성공 여부 |
| `output` | **모델 원본 출력. 가공하지 않음** |
| `error` | 실패 시 메시지 |
| `elapsed` | 장당 소요 초 |

25장마다 flush하며, 재시작 시 처리 완료 건은 건너뛴다.

---

## 8. 데이터 연결

### 경로

```
/kaggle/input/datasets/osamahosamabdellatif/
  high-quality-invoice-images-for-ocr/batch_1/batch_1/batch1_1/batch1-0182.jpg
```

초기에 `/kaggle/input/high-quality-.../` 로 잡아 **이미지 0장**이 나왔다.
`/kaggle/input/datasets/<owner>/` 한 겹이 더 있다.

구조 의존을 없애기 위해 `/kaggle/input/**/batch1-*.jpg` 로 탐색하고,
`batch_3` 경로는 제외한다(batch_1·2의 복사본이 들어 있음).

### 매칭 결과

```
실행 목록 1,413건
batch1 이미지 1,489장
경로 확인 1,413 / 누락 0
```

**로컬 정답지와 Kaggle 이미지가 완전히 일치한다.**

### 업로드한 것

| 파일 | 용도 |
|---|---|
| `run_list.csv` | Kaggle Dataset `trade-doc-runlist` |

**정답지(`truth_*.csv`)는 올리지 않았다.**
실행 환경에서 정답을 참조할 여지를 없애기 위함이다.

---

## 9. README 반영 문구

> Granite Docling은 프롬프트에 따라 표 추출 결과가 크게 달라졌다.
> `"Convert this page to docling."`에서는 표 셀이 0개로 총액을 읽지 못했고,
> `"Convert this page to OTSL."`에서 40셀이 정상 추출됐다.
> 5개 프롬프트를 비교해 최선값을 채택했으며, PaddleOCR에도 동일한 수준의
> 설정 탐색을 수행했다.
>
> float16에서는 출력이 깨져 사용할 수 없었고, bfloat16과 float32는
> 토큰 단위까지 동일한 출력을 냈다. bfloat16을 채택했다.
>
> Kaggle 무료 GPU 시간 제약으로 1,413장 중 300장을 사용했다.
> 실행 목록은 금액 5분위로 층화 후 셔플되어 있어 상위 300건이
> 전 금액대를 고르게 포함한다.

---

## 10. 다음 단계

3단계에서 **같은 300건, 같은 순서**로 PaddleOCR-VL-1.6을 실행한다.
설정 탐색도 동일한 수준으로 수행한다.