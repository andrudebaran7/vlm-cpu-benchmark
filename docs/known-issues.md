# Known issues & reproducibility notes

This log records practical inconveniences encountered while running the
benchmark on real hardware. It is intended both as engineering
documentation and as source material for the "lessons learned /
reproducibility" discussion in the accompanying paper.

Each entry: symptom → root cause → resolution → status.

---

## I1 — `AutoModelForVision2Seq` removed in `transformers` 5.x

- **Symptom:** running `smolvlm-256m` (fp32) failed immediately with
  `ImportError: cannot import name 'AutoModelForVision2Seq' from
  'transformers'`.
- **Environment:** `transformers==5.14.1`.
- **Root cause:** the `AutoModelForVision2Seq` auto class was renamed to
  `AutoModelForImageTextToText` in `transformers` 5.x. The old name is no
  longer exported.
- **Resolution:** `src/vlmbench/models/smolvlm.py` now imports
  `AutoModelForImageTextToText` with a fallback to the old name for
  older `transformers` installs. The same rename affects
  `optimum.onnxruntime.ORTModelForVision2Seq` used by the onnx-int8
  path (`backends/onnx_export.py`, `smolvlm.py`) — not yet exercised, see
  I4.
- **Status:** RESOLVED (fp32 path).

## I2 — CPU energy (RAPL) not measurable without privileges

- **Symptom:** `energy_j` column is empty for every cell.
- **Root cause:** Intel RAPL counters (`/sys/class/powercap/...` or perf
  energy events) are not readable by an unprivileged user on this
  machine, so the energy profiler yields no reading.
- **Resolution options (undecided):** (a) run the benchmark with
  elevated privileges / adjusted `perf_event_paranoid`; or (b) drop the
  energy dimension from the paper and report latency + peak RSS only.
- **Status:** OPEN — decision pending.

## I3 — First real cell was silently skipped on re-run (resume dedup)

- **Symptom:** after fixing I1, re-running produced `wrote 0 cells` and an
  empty results table.
- **Root cause:** `run_matrix` treats any cell already present in
  `results.jsonl` — including `failed` cells from a previous run — as
  "completed" and skips it (resume/idempotency behaviour). The failed
  cell from the pre-fix run masked the fixed re-run.
- **Resolution:** delete (or use a fresh) `results/` output directory
  when re-running after a fix. Candidate improvement: exclude `failed`
  cells from `completed_keys()` so failures are retried automatically.
- **Status:** WORKAROUND (manual); code improvement candidate.

## I4 — onnx-int8 backend dependencies not installed

- **Symptom:** the `onnx-int8` backend cannot run.
- **Root cause:** it requires the `quant` extra (`onnxruntime`,
  `optimum`, ...) which is not installed; and its export path uses the
  renamed `ORTModelForVision2Seq` (see I1), so a version fix will likely
  be needed there too.
- **Status:** OPEN — deferred; scaling runs use `fp32` only for now.

## I5 — `trust_remote_code` VLMs break under `transformers` 5.x

- **Symptom:** scaling the fp32 run to all four models, three of four
  failed at load/inference time while only SmolVLM (a native
  `transformers` architecture) succeeded:
  - `moondream2`: `AttributeError: 'HfMoondream' object has no attribute
    'all_tied_weights_keys'`
  - `internvl2_5-2b`: `AttributeError: 'InternVLChatModel' object has no
    attribute 'all_tied_weights_keys'`
  - `florence2-base`: `AttributeError: 'Florence2LanguageConfig' object
    has no attribute 'forced_bos_token_id'`
- **Environment:** `transformers==5.14.1`.
- **Root cause:** these models ship their modelling code on the Hub and
  are loaded with `trust_remote_code=True`. That remote code was written
  against `transformers` 4.4x. `transformers` 5.x changed internal
  contracts the remote code does not satisfy — the base class now expects
  `all_tied_weights_keys` on every model, and Florence-2's config lacks
  `forced_bos_token_id` that the newer generation path reads. A related
  deprecation is already visible in the logs: *"From v4.50 onwards,
  `PreTrainedModel` will NOT inherit from `GenerationMixin`"*. Only the
  SmolVLM path, which uses a first-class `transformers` auto class, is
  insulated.
- **Paper relevance:** this is a concrete, quantifiable reproducibility
  hazard — a single major dependency bump silently disabled 75% of the
  model suite. Worth reporting as a finding, not just a footnote.
- **Resolution options:** (a) pin `transformers` to a 4.4x line
  (candidate: 4.49.x — still has `AutoModelForImageTextToText` for
  SmolVLM and still inherits `GenerationMixin`) so all four models share
  one environment; (b) pin a compatible remote-code revision per model;
  (c) vendor/patch the remote code. Option (a) is the most reproducible.
- **Status:** RESOLVED via option (a): environment pinned to
  `transformers==4.49.0` (also downgrades `tokenizers` to 0.21.x and
  `huggingface-hub` to 0.36.x). Under 4.49 moondream2 and florence2-base
  load and run; two smaller follow-on issues surfaced (I6, I7).

## I6 — `from_pretrained` dtype kwarg renamed (`dtype` vs `torch_dtype`)

- **Symptom:** after pinning `transformers==4.49.0`, SmolVLM failed with
  `TypeError: Idefics3ForConditionalGeneration.__init__() got an
  unexpected keyword argument 'dtype'`.
- **Root cause:** the I1 fix passed `dtype=torch.float32`, but that kwarg
  name is `transformers` 5.x. In 4.x the argument is `torch_dtype`.
- **Resolution:** `smolvlm.py` now tries `dtype=` and falls back to
  `torch_dtype=` on `TypeError`, so it works on both major versions.
- **Status:** RESOLVED.

## I7 — InternVL2.5 requires `sentencepiece` (not pulled in transitively)

- **Symptom:** `internvl2_5-2b` failed with
  `ModuleNotFoundError: No module named 'sentencepiece'`.
- **Root cause:** InternVL2.5's tokenizer needs `sentencepiece`, which is
  not a declared dependency of the `[models]` extra nor pulled in by
  `transformers` 4.49 by default.
- **Resolution:** install `sentencepiece` (added to the `[models]`
  extra).
- **Status:** RESOLVED.

## I8 — moondream2 is ~20x slower than the other small VLMs on CPU

- **Observation:** at fp32 on CPU, per-inference latency was ~84 s for
  moondream2 vs ~4 s (florence2-base) and ~18 s (smolvlm-256m). Peak RSS
  ~4.8 GB. Not a bug, but a material efficiency result worth surfacing in
  the paper: nominal parameter count is a poor predictor of CPU latency.
- **Status:** NOTED (result, not a defect).

## I9 — InternVL2.5 tokenizer also requires `protobuf`

- **Symptom:** after installing `sentencepiece` (I7), `internvl2_5-2b`
  still failed. The InternLM2 tokenizer first raised
  `RuntimeError: INTERNAL: piece must not include null character` from
  the SentencePiece fast path, then fell back to a protobuf-based loader
  and raised `ImportError: ... requires the protobuf library`.
- **Root cause:** the tokenizer's fallback conversion path needs
  `protobuf`, another undeclared transitive dependency.
- **Resolution:** install `protobuf` (added to the `[models]` extra).
- **Status:** RESOLVED.

## I10 — InternVL2.5 tokenizer loads as `bool` under transformers 4.49 (unresolved)

- **Symptom:** with `protobuf` present, `internvl2_5-2b` fails inside the
  remote `chat()` at
  `tokenizer.convert_tokens_to_ids(...)` with
  `AttributeError: 'bool' object has no attribute 'convert_tokens_to_ids'`.
- **Diagnosis:** `AutoTokenizer.from_pretrained('OpenGVLab/InternVL2_5-2B',
  trust_remote_code=True)` **returns the Python value `False`** instead of
  a tokenizer object (confirmed by printing `type(...)` → `bool`), for
  both `use_fast=True` and `use_fast=False`. Upstream, SentencePiece
  0.2.x raises `RuntimeError: INTERNAL: piece must not include null
  character` while loading InternLM2's `tokenizer.model`; transformers
  4.49's `_from_pretrained` catches this on its protobuf-decode-error
  path and ends up returning `False` rather than raising.
- **Attempts that did NOT fix it:** installing `protobuf`; `use_fast=True`
  vs `False`; downgrading `sentencepiece` to 0.1.99 (fails to build on
  Python 3.12).
- **Candidate resolutions (not yet attempted, each has a cost/risk):**
  (a) a different `transformers` version that both parses this tokenizer
  and keeps moondream2/florence2 working; (b) pin an older InternVL model
  revision whose tokenizer code predates the incompatibility; (c) build
  the tokenizer manually from the raw SentencePiece model, bypassing the
  broken `from_pretrained` path; (d) run InternVL in a separate, isolated
  environment.
- **Paper relevance:** the strongest reproducibility finding so far — a
  widely-cited 2B VLM cannot be loaded at all in an environment where two
  other community VLMs work, due to a silent `from_pretrained` failure
  that returns a wrong-typed object instead of erroring.
- **Status:** OPEN — 3 of 4 models currently run; InternVL deferred
  pending a decision on how much effort to invest.

---

### Dependency summary for a working 3-of-4-model fp32 run

Beyond `pip install -e '.[models]'`, a clean run required pinning
`transformers==4.49.0` and adding `sentencepiece` and `protobuf`. The
net reproducibility lesson: three of the four community VLMs are highly
sensitive to the `transformers` major version and carry undeclared
tokenizer dependencies — a non-trivial environment-assembly cost that a
"just pip install the model" narrative hides.
