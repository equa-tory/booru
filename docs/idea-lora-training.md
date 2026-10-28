# Idea (not built): train a LoRA from selected posts

Status: discussed 2026-10-29, nothing built. Next step = a short benchmark (below) before any UI.

## Why it looks feasible
- ComfyUI 0.32 on this host already has native training nodes: `TrainLoraNode` (inputs: model, latents,
  positive, batch_size, grad_accumulation_steps, steps, learning_rate, rank, optimizer, loss_function, seed,
  training_dtype, lora_dtype, quantized_backward, algorithm, gradient_checkpointing, checkpoint_depth,
  offloading, existing_lora, bucket_mode, bypass_mode), `SaveLoRA`, `MakeTrainingDataset` (images, vae, clip,
  texts), `Save/LoadTrainingDataset`, `LoadImageTextDataSetFromFolder`. No new software, base = the current
  checkpoint (WAI-Illustrious fp8).
- Captions are free: the posts are already Danbooru-tagged (the checkpoint's own language). Standard practice:
  drop the tags that describe the shared concept so the LoRA learns it under the trigger word.

## Planned flow
1. Gallery selection → "🎓 train LoRA": name, trigger word, kind (character / object / style).
2. Dataset: the selected stills (resize/bucket), captions = trigger + post tags minus the concept tags.
3. Job in the AI queue (`_start_comfy_job`, kind e.g. `lora_train`), progress + ■ stop; workflow built in code
   like `build_img2img` (MakeTrainingDataset → TrainLoraNode → SaveLoRA).
4. Output into ComfyUI's loras folder, e.g. `loras/booru/<name>.safetensors` (+ metadata with the trigger word
   so `lora_catalog` / the picker show it as a trigger group) → usable on ✨ create (txt2img and painted areas).

## Expectations on the Tesla P4 (estimates, NOT measured)
- No fast fp16/bf16 → fp32 compute: guess 5-12 s per step (0.4-0.8 MP), 1,000-2,000 steps → 2-6 h, overnight.
- 8 GB VRAM: tight; fp8 base + gradient checkpointing + offloading should fit.
- Heat: hours of full load on a passively cooled card. The queue's °C gate only delays the START; training
  needs its own pause-when-hot between steps.
- Data: 15-40 varied images for a character/object, 50+ for a style. Photos on an anime base = weaker.

## First step when picking this up
A 50-step test training on ~10 posts: measure s/step, peak VRAM, temperature; then decide defaults
(steps, rank, resolution) and build the button + job.
