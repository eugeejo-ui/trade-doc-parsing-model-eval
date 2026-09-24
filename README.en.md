<div align="right">

[한국어](README.md)

</div>

# Which Model for Which Customer?

### Trade Finance Document Parsing — Granite Docling vs PaddleOCR-VL

Two document parsing models were measured under identical conditions for reading
letter-of-credit documents on a bank's internal servers, establishing a measured
decision baseline for **which customer situation suits which model**.

Conclusion: the two models performed identically, and the factor driving the
choice was found not to be the model.

---

## Conclusion

**Axis 1 — Performance**

Across 150 structured commercial invoices, extraction accuracy for amount, date,
and seller name was found to be identical between the two models.

When the scope was extended to full-page text, PaddleOCR-VL was measured 4.18pp
ahead in numeric recall. However, the items Granite omitted were confirmed to be
customer addresses and Tax IDs, which do not fall among letter-of-credit
comparison fields.

**Axis 2 — Deployment Burden**

Granite Docling 258M (515MB) and PaddleOCR-VL 0.9B (1.92GB) differ by a factor
of 3.5 in model size.

Processing speed was found to favor PaddleOCR (18.0s vs 26.3s on T4). The design
phase anticipated that PaddleOCR would require multiple calls per page due to
lacking full-page parsing support; measurement confirmed that a single call was
sufficient.

**Judgment**

> For field-level extraction, 258M appears sufficient in performance terms.
> For full-page indexing or position-based post-processing, 0.9B is expected to
> deliver better performance.
>
> Adoption was found to be determined less by model performance than by
> **how many resubmissions the monthly operating cost corresponds to**.

---

## What Factors Determine Adoption

| Factor | Effect on break-even |
|---|---|
| **Operating cost / resubmission cost ratio** | 10× → 15 docs, 100× → 150 docs |
| **Cost of one resubmission** | 4× → required volume falls to 1/4 |
| Model detection rate | 20pp drop → 4 doc difference |
| Model false positive rate | 20× increase → **no change** |

Cost structure was found to exert greater influence on the decision than model
performance.

### Break-even

| Monthly operating cost (= M resubmissions) | Break-even |
|---|---|
| 5 | 8 docs |
| 10 | 15 docs |
| 50 | 75 docs |
| 100 | 150 docs |

It should be noted that the cost of one resubmission includes not only rework
labor but the opportunity cost of up to five banking days of re-examination delay
(UCP 600). Break-even was confirmed to fall lower for small exporters where a
five-day delay is critical to cash flow.

The design phase anticipated that "false positive review burden would offset the
savings"; it was confirmed that raising the false positive rate 20× (2.5% → 50%)
did not move the break-even point. This was attributed to false positive review
cost being orders of magnitude smaller than resubmission cost. The factor
offsetting the savings was found to be **fixed operating cost**, not false positives.

---

## Judgment Procedure

Rules were validated first, and model measurement was conducted using the
validated rules.

**1. Prior verification of rule correctness**

The amount-matching rule was validated against 8,564 real cases from BPI
Challenge 2019. Rule output matched the definitional ground truth 100%, and
varying the tolerance from 0.001 to 10 EUR was confirmed not to change any verdict.

**2. Prior fixing of ground truth**

Nine condition-specific ground truth files were generated and written to disk
before any model was run. It should be noted that this was a measure to eliminate
any room for adjusting answers after observing results.

**3. Model execution under identical conditions**

The same 150 images were run in the same order.
Five prompts were compared for each model and the best adopted.

**4. Scoring through a single extractor**

It should be noted that writing separate parsers per model would introduce parser
quality differences into the results.

**5. Application of validated rules and conversion to detection rate**

**6. Substitution of detection rate into the cost model**

---

## Method of Ensuring Fairness

**A single extractor** was applied to both models' outputs.

It should be noted that `Table Recognition:` was selected as the PaddleOCR prompt
in order to obtain the same OTSL tag structure as Granite. Output format, not
accuracy, was taken as the selection criterion.

Granite's table cell extraction was found to vary between 0 and 40 cells
depending on prompt. `Convert this page to docling.` produced empty table shells,
while 40 cells were confirmed to be filled under `Convert this page to OTSL.`

### Extractor revision log

| Issue | Affected | Fix |
|---|---|---|
| Postal code mistaken for amount | Granite | Require two decimal places |
| Military addresses not truncated | Both, 43 cases | Add address keywords |
| `Seller:` and name in a single cell | PaddleOCR | Add prefix handling path |

No rule that would disadvantage the verdict of only one of the two models was
adopted. Score changes for both models were recorded after each revision.

| Stage | Granite (name) | PaddleOCR (name) |
|---|---|---|
| Initial | 84.0% | 85.3% |
| After address keywords | 98.0% | 96.0% |
| After Seller prefix handling | **100.0%** | **100.0%** |

---

## Results

### Field extraction accuracy (n=150)

| | Amount | Date | Seller |
|---|---|---|---|
| Granite Docling | 98.0% | 100.0% | 100.0% |
| PaddleOCR-VL | 98.0% | 100.0% | 100.0% |

The three amount errors produced identical values across both models. In those
cases the ground truth JSON `total` recorded the net rather than gross amount;
this was confirmed to be a dataset notation inconsistency rather than a model
error. Ground truth was left unmodified.

### Defect detection

Detection rate 100% and false positive rate 2.5% were identical across both
models, with a 0.0pp difference in all nine conditions.

Scaling the injected discrepancy 500× (0.1% → 50%) did not change the detection
rate. Deadline conditions were identical from 1 to 30 days. Detection rate was
found to be determined by parsing accuracy rather than defect magnitude.

### Full-page recognition (secondary metric)

| | Numeric recall (order-agnostic) |
|---|---|
| Granite | 79.06% |
| PaddleOCR | 83.25% |

Initial measurement showed a 17.85pp gap, of which 13.67pp was confirmed to
originate from **reading order**. It should be noted that Granite places the
customer address block at the end of its output whereas the ground truth text
follows visual reading order, a structure that disadvantages Granite in
order-sensitive comparison. Re-measurement with order-agnostic multiset
comparison was confirmed to reduce the gap to 4.18pp.

The items Granite omitted were Tax IDs, postal codes, and street numbers, all
within the customer address region. No omissions were observed in amounts,
quantities, or dates.

---

## Assumptions

- Customer: banks handling letters of credit
  (under Korean foreign exchange law, securities firms may not handle
  payment or collection)

- **Deployment: bank internal servers, not external API calls**

  Since April 2026, SaaS services that have passed Financial Security Institute
  evaluation have been permitted on internal business networks, but this is
  limited to evaluated services and designated financial institutions.
  Furthermore, as the BPI and Pronto cases illustrate, restrictions on exporting
  sensitive data externally are in some cases set by institutions themselves,
  independently of regulation.

- Subject: parsing base models, not finished systems
- Comparison: manual review (AS-IS) versus AI adoption (TO-BE)
- Out of scope: replacing an existing solution where the task is already automated

---

## Limitations

**Data**

- Synthetic, structured invoices; handwriting, scan noise, and irregular layouts
  are not included
- General commercial invoices; it should be noted that these are not
  letter-of-credit commercial invoices
- 150 samples are insufficient to distinguish a 1–2pp difference
- Amount notation was confirmed to mix period decimals (1,078 cases) and
  comma decimals (336 cases)

**Defect rate**

- It should be noted that the 70% reported by ICC is the
  **first-presentation rejection rate**, not the final refusal rate
- It should be noted that this is a figure measured after human review has
  already occurred

**Condition synthesis**

- No shipment date field exists, so issue date was used as the comparison target
- It should be noted that deadlines were synthesized using the 21-day
  presentation period of UCP 600 Article 14(c)
- It should be noted that UCP 600 strict compliance treats even a one-day or
  one-cent difference as a discrepancy; relaxed conditions were applied in order
  to isolate parsing accuracy

**Cost model**

- Absolute figures for resubmission cost and operating cost cannot be confirmed,
  as no public data exists
- It should be noted that reducing them to a ratio does not remove the uncertainty
- A high false positive rate may lead staff to distrust and discontinue use of
  the system; it should be noted that this is an adoption issue rather than a
  cost one and is therefore not reflected in the formula

**Rule validation**

- Only the amount rule was validated against external real data
- It should be noted that deadline and name rules could not be validated,
  as no corresponding fields exist in BPI 2019

**Execution environment**

- Single Tesla T4 GPU, no batching applied
- Absolute timings differ on A100/H100

---

## Reproduction

```bash
pip install -r requirements.txt

python src/run_bpi_check.py      # Stage 0: rule validation
python src/inspect_kaggle.py     # Stage 1: schema inspection
python src/make_truth.py         # Stage 1.5: ground truth generation
# Stages 2–3: Kaggle notebooks (see notebooks/)
python src/score.py              # Stage 4: scoring
python src/detect.py             # Stage 5: detection rate
python src/text_accuracy.py      # Stage 5b: text accuracy
python src/breakeven.py          # Stage 6: break-even
```

It should be noted that source data is not included due to size; see sources below.
Per-stage design notes and rationale are documented in [`docs/`](docs/) (Korean).

---

## Sources

- **Invoice data**: Osama Hosam Abdellatif, *High Quality Invoice Images for OCR*, Kaggle
- **Process log**: van Dongen, B.F., *Dataset BPI Challenge 2019*,
  4TU.Centre for Research Data,
  doi:10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- **Defect rate**: Dave Meynell, Senior Technical Advisor, ICC Banking Commission —
  65–80% first-presentation rejection rate for letters of credit
- **Presentation period**: UCP 600 Article 14(c)
- **Models**: `ibm-granite/granite-docling-258M`, `PaddlePaddle/PaddleOCR-VL-1.6`
  (both Apache 2.0)