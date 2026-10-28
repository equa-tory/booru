# Idea (not built): PixAI tagger v1.0 as the character model

Status: found 2026-10-29, nothing built.

- Models never update in place: a new version is a new Hugging Face repo, often with a new format,
  resolution, vocabulary and thresholds — so "check + pull update" means adding the new version as an
  option (code + test), not a generic download button.
- `pixai-labs/pixai-tagger-v1.0` (released 2026-09-15): 30,877 tags, **8,308 characters** (v0.9 ONNX we use:
  3,720 characters), data cutoff May 2026 (v0.9: Jan 2025), +10.4 pts character F1 / +6.8 general F1 vs v0.9.
  Default thresholds: character 0.27, general 0.17.
- Cost: input 1008×1008 instead of 448×448 (~5× pixels) → expect several times slower than v0.9's ~0.3 s/img
  on the Tesla P4, more VRAM. Community ONNX exports exist (e.g. `noaione/pixai-tagger-v1.0-onnx`,
  `yasamari/pixai-tagger-v1.0-onnx`) — pick one, verify outputs against the official model first.
- Plan: add as `pixai1` in `ai_models.py` beside `pixai`, own preprocessing (check its preprocess config),
  benchmark s/img on the P4, then offer "re-tag characters with v1.0" (resumable Task, like `char_retag`).
