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
- **Status:** RESOLVED. Root cause pinned down: InternLM2's
  `tokenizer.model` contains exactly one degenerate piece (index 354,
  type UNKNOWN, score ~-188) whose string is a literal NUL character.
  SentencePiece 0.2.x rejects any model with a NUL in a piece ("piece
  must not include null character"); transformers 4.x catches the
  resulting `RuntimeError` and returns `False` (assuming a non-SPM file
  with a fast/tiktoken fallback that InternLM2 does not have). The proper
  NUL byte is already representable via the BYTE token `<0x00>` (id 3), so
  the degenerate piece is safe to sanitize. `models/internvl2_5.py` now
  loads the tokenizer from a patched snapshot: it replaces the NUL in the
  offending piece with `␀` (U+2400), caches the patched `tokenizer.model`
  (plus symlinks to the other tokenizer files) under the vlmbench cache,
  and reuses it on later runs. Verified end-to-end: InternVL2.5-2B now
  loads and runs a real DocVQA inference. All four models load.

## I11 — Florence-2 returns captions, not answers, under a plain question prompt

- **Symptom:** on the DocVQA run, `florence2-base` scored ANLS 0.000; its
  outputs are descriptive captions ("The image is an advertisement
  for...") rather than extracted answers.
- **Root cause:** Florence-2 is task-token driven (`<OCR>`, `<CAPTION>`,
  `<VQA>`, ...); the adapter sends the raw question with no task token, so
  the model defaults to captioning.
- **Resolution:** not applied — flagged as an adapter-prompting
  limitation. Candidate fix: prepend an appropriate Florence-2 task token
  and parse the structured output.
- **Status:** OPEN — reported as measured; adapter improvement candidate.

## I12 — DocVQA latency far exceeds the synthetic-image estimate

- **Observation:** on real high-resolution DocVQA document images
  (~1370x1480), mean per-inference latency (fp32, CPU, N=100) was
  Florence-2 ~4.2 s, SmolVLM-256M ~23.3 s, and Moondream2 ~215 s — the
  last ~2.5x higher than the ~85 s seen on the 64x64 synthetic `sample`
  image (I8). High-resolution documents inflate the visual token budget;
  the full 4-model run took ~8 h wall-clock, dominated by Moondream2.
- **Planning lesson:** never extrapolate CPU runtime from a toy image;
  size the subsample against the real task's images.
- **Status:** NOTED (result + planning lesson).

## I13 — Quantization tooling for small VLMs is immature (ONNX/optimum)

- **Symptom:** the implemented `onnx-int8` path (optimum export of
  SmolVLM) does not work, and the model's own pre-exported ONNX cannot be
  loaded by optimum either.
- **Details:**
  - `ORTModelForVision2Seq.from_pretrained(..., export=True)` fails:
    `ValueError: Trying to export a idefics3 model, that is a custom or
    unsupported architecture` — optimum has no ONNX exporter for
    Idefics3/SmolVLM.
  - SmolVLM-256M *does* ship pre-quantized ONNX on the Hub, but in the
    Transformers.js layout — three separate graphs (`vision_encoder`,
    `embed_tokens`, `decoder_model_merged`, each with `_int8`/`_q4`/... 
    variants) — not optimum's `encoder_model.onnx`/`decoder_model.onnx`
    seq2seq layout. `ORTModelForVision2Seq` cannot consume it
    (`Idefics3Config has no attribute num_attention_heads`).
- **Consequence:** running int8 requires a bespoke onnxruntime inference
  loop (vision encoder → token embeddings → autoregressive decoder with
  KV cache across three sessions), i.e. real code, not a config switch.
- **Status:** custom ONNX-INT8 backend under implementation (see
  `backends/onnx_smolvlm.py`).

## I14 — moondream2 advertised GGUF backends it does not implement

- **Symptom:** `moondream2` declared `gguf-q4`/`gguf-q8` in
  `supported_backends`, but its `load()` ignored the `backend` argument
  and always built the fp32 transformers model. A gguf run would have
  produced fp32 numbers silently mislabeled as quantized.
- **Resolution:** removed the unimplemented backends from the model's
  advertised set (so the orchestrator records them as UNSUPPORTED) and
  added a guard in `load()` that raises for any non-fp32 backend.
- **Status:** RESOLVED (honesty fix). A real GGUF path via
  `llama-cpp-python` remains possible future work.

## I15 — SmolVLM fp32 adapter echoed the prompt, forcing ANLS to 0

- **Symptom:** SmolVLM-256M scored ANLS 0.000 on the fp32 DocVQA run. When
  the ONNX-INT8 backend (which decodes only newly generated tokens) began
  returning coherent, sometimes-correct answers to the same questions, the
  fp32 zero became suspect.
- **Root cause:** the fp32 adapter decoded the *entire* `generate` output
  with `batch_decode(ids)`, i.e. the chat prompt plus the answer, yielding
  strings like ``"User:\n...Assistant: The total amount ... is $840.00."``.
  ANLS against a bare gold answer is then ~0 regardless of whether the
  model was right, so the fp32 score reflected an output-extraction bug,
  not (only) model capability.
- **Resolution:** slice off the prompt tokens (`ids[:, input_len:]`) before
  decoding, matching the ONNX path. fp32 SmolVLM must be re-run for a fair
  number. Like I11 (Florence captioning), this is a reminder that naive
  adapters can understate a model's measured accuracy.
- **Status:** RESOLVED (adapter fixed; re-run required).

---

### Dependency summary for a working 4-model fp32 run

Beyond `pip install -e '.[models]'`, a clean run required pinning
`transformers==4.49.0`, adding `sentencepiece` and `protobuf`, and
patching InternVL2.5's tokenizer (I10). All four community VLMs are highly
sensitive to the `transformers` major version, carry undeclared tokenizer
dependencies, and — in one case — ship a corrupt tokenizer token; a
non-trivial environment-assembly cost that a "just pip install the model"
narrative hides.
