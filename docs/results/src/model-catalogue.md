# Open-weight LLM families for cross-family subliminal-learning tests (checked 2026-11-05)

**How this was checked.** I ran four parallel research agents and covered Qwen myself.
- **Sources:** model metadata (size, creation date, license, gating) comes from the Hugging Face API. Chat templates come from each repo's raw `tokenizer_config.json` or `chat_template.jinja` on `main`.
- **Template column:** every "Default system prompt" cell comes from rendering the template with jinja2 on a single user turn (`add_generation_prompt=True`), not from model cards.
- **vLLM:** checked against the vLLM v0.32.0 model registry (tag dated 2026-10-18).
- **transformers:** checked against the transformers source on main (~v5.22).
- **Gated repos:** templates were read from ungated mirrors (unsloth, mlx-community), as noted in the cells.
- **Re-verified by me:** I re-rendered the templates of every shortlist model. I also confirmed that about 30 of the more surprising post-cutoff repos exist via the HF API (e.g. Muse Glimmer, Gemma 4, Qwen3.8, DeepSeek V4, Granite 4.2).

**Legend**
- **Release** is the month of public release. Where only the repo creation date is known, that date is used.
- **Fit on 1×L40S:**
  - "yes" means bf16 weights plus LoRA fit on one 48 GB card. Weights take about 2 bytes per parameter, and for MoE models the total parameter count is what matters.
  - "QLoRA/multi" means the model needs QLoRA or several GPUs.
  - "no" means a very large MoE.
- **"Adds nothing"** means the template renders no system turn when the conversation has no system message.
- **"(?)"** marks a cell I could not verify.

**Cross-cutting findings that matter for this project**
- **Teacher and student templates must match token for token.** Many templates stamp *today's date* into the prompt, so the same prompt renders differently from day to day. These are Llama 3.2, SmolLM3, Apertus 2509, Granite 3.1–3.3, Mistral Small 3/3.1 and gpt-oss.
- **Thinking is on by default** in many 2025–26 releases: Nemotron Nano v2 / 3 / 3.5, Qwen3.5+, Granite 4.2, Gemma 4 when enabled, GLM-4.7-Flash and LFM2.5/3. Teacher number sequences will come with `<think>` traces unless you turn thinking off (`enable_thinking=False`, `/no_think`, or the model's equivalent).
- **Mistral hazard: vLLM and HF render prompts differently.** vLLM v0.32 `tokenizer_mode=auto` switches every `mistralai` repo to mistral_common, which inserts *no* default system prompt. HF `AutoTokenizer` with the repo's jinja template inserts a long "Le Chat" prompt for Ministral 3, Small 3/3.1/4 and Large 3. Your vLLM teacher and HF student would then see different prompts.
- **Qwen in disguise:** several non-Qwen-branded models are Qwen underneath. These are DeepSeek-R1-Distill-Qwen-*, R1-0528-Qwen3-8B, Step3-VL-10B (Qwen3-8B decoder) and MiMo-V2.6-Distill-Qwen-9B. MiMo-7B was pretrained from scratch but uses the Qwen2 tokenizer and a Qwen-style template.

---

## 1. Llama (Meta), plus Meta's newer "Muse" line

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit 1×L40S | Data openness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.1 8B / 70B / 405B | 8.0 / 70.6 / 406B | 7/2024 | Yes | Instruct; no reasoning | FP8 for 405B (community: ubiquitous) | Dense, 128K | Llama 3.1 license; gated (manual) | native / native | **Inserts fixed** `Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024`. A user system message is appended after it in the same turn. Verified on the 8B via the unsloth mirror; 70B/405B assumed the same (?) | 8B yes; 70B QLoRA/multi; 405B no | Closed (~15T tokens) |
| Llama 3.2 1B / 3B | 1.24 / 3.21B | 9/2024 | Yes | Instruct | SpinQuant/QLoRA INT4 for ExecuTorch (ubiquitous) | Dense, 128K | Llama 3.2; gated | native / native | Same block, but `Today Date` = **current date** (changes daily; pin it with `date_string`) | yes | Closed |
| Llama 3.2 11B / 90B Vision | 10.7 / ~88B (?) | 9/2024 | Yes | Instruct | (community) | Dense + cross-attention vision | Llama 3.2; gated | native / **removed from vLLM in v0.10.2** | Same as 3.2 | 11B yes; 90B no | Closed |
| Llama 3.3 70B | 70.6B | 12/2024 | No (uses the 3.1 70B base) | Instruct | (community) | Dense, 128K | Llama 3.3; gated | native / native | Same fixed "26 Jul 2024" block as 3.1 | QLoRA/multi | Closed |
| Llama 4 Scout 17B-16E | 109B / 17B active | 4/2025 | Yes | Instruct | BF16 only (int4 on the fly) | MoE, multimodal, 10M | Llama 4; gated | native / native | **Adds nothing** (a tool prompt appears only when tools are passed) | no | Closed (~40T) |
| Llama 4 Maverick 17B-128E | 400B / 17B | 4/2025 | Yes | Instruct | FP8 | MoE, 1M | Llama 4; gated | native / native | Adds nothing | no | Closed |
| Muse Glimmer 30B (`meta-models`) | 29.8B dense (incl. ~1.8B vision) | 8/2026 | **No** | Reasoning-only; effort low…xhigh (default high); no off switch | GGUF, ExecuTorch (community FP8/NVFP4/AWQ/bnb) | Dense, 3 local : 1 global attention, 131K, text+image | **Apache-2.0; ungated** | native (transformers ≈5.15 (?)) / native | **Inserts** "You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: \<today>.\n\nReasoning strength: high.\n\n# Valid recipients…". A user system message replaces the identity and date lines, but the effort and recipients lines remain | QLoRA/multi (~60 GB) | Closed |
| Muse Spark Open 816B-A42B | 822B / 42B | 10/2026 | No | Reasoning-only | NVFP4 | MoE, 1M, multimodal | Muse Model License; gated | native / native (?) | Always prepends identity + cutoff + date, then "Reasoning strength: 512." | no | Closed |

**Left out:** Llama 3.0, Guard/Prompt-Guard, CodeLlama and FAIR MobileMoE. Web reports of "Llama 4.5" and "Llama 5" have no HF repos; I treat them as **unverified**.

**Notes**
- **Tokenizer:** base and instruct share it, but the base eos is `<|end_of_text|>` and the base has no chat template.
- **Avoid this mirror:** `NousResearch/Meta-Llama-3.1-8B-Instruct` carries a stale template with no date block.

## 2. NVIDIA Nemotron

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data openness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-Nemotron-Nano-8B-v1 | 8.0B | 3/2025 | No own base (from Llama-3.1-8B-Instruct) | Hybrid; "detailed thinking on/off" in the system prompt | none (community) | Llama dense, 128K | NVIDIA Open Model License + Llama 3.1; ungated | native / native | **Empty system header**, no text | yes | Post-training data open (CC-BY-4.0); pretraining is Meta's |
| Llama-3.1-Nemotron-Nano-4B-v1.1 | 4.5B | 5/2025 | Yes (Minitron-4B-Width-Base) | Same toggle | — | Llama, 128K | Same | native / native | **Inserts `detailed thinking off`** | yes | Same |
| Llama-3.3-Nemotron-Super-49B v1 / v1.5 | 49.9B | 3/2025, 7/2025 | No | v1.5 thinking on by default; `/no_think` turns it off | FP8, NVFP4 | DeciLM (NAS-pruned), 128K | Same; ungated | trust_remote_code / native | Empty system header | QLoRA/multi | Post-training open |
| Llama-3.1-Nemotron-Ultra-253B | 253B | 4/2025 | No | Toggle | FP8 | DeciLM | Same | remote / native | **Inserts `detailed thinking on`** | no | Post-training open |
| Nemotron-H 4B / 8B / 47B / 56B | 4.5 / 8.1 / 46.8 / 56B (?) | 3–6/2025 | Yes (`-Base-8K`) | 4B-Instruct; 8B/47B-Reasoning | FP8 (Reasoning models) | Mamba-2 hybrid; 8K base / 128K | **Research-only NVIDIA license**; ungated | remote code (native `nemotron_h` ≥5.3 (?)) / native | Empty system turn; reasoning toggled by the literal `{'reasoning': True/False}` in the system message | 4B/8B yes; 47B/56B QLoRA/multi | (?) |
| **Nemotron-Nano-9B-v2 / 12B-v2** | 8.9 / 12.3B | 8/2025 | **Yes (both)** | Hybrid reasoning, **on by default**; `/think` and `/no_think` work in system or user messages | 9B: FP8, NVFP4; 12B: none | Mamba-2 hybrid (mostly Mamba, a few attention layers), 128K | NVIDIA Open Model License; ungated | remote or native `nemotron_h` / native | **Empty system turn** `<SPECIAL_10>System\n\n`; forces `<think>\n` (verified) | yes (12B nearer the edge) | **Open pretraining data** (Nemotron-CC-v2, CC-Math, Pretraining-Code/SFT; 20T tokens) plus post-training v1/v2 |
| Nemotron 3 Nano 30B-A3B | 31.6B / ~3.5B | 12/2025 | Yes | `enable_thinking` (default on) | FP8, NVFP4 (GGUF) | Mamba-2 + MoE + 6 attention layers; 256K (to 1M) | Nemotron Open Model License; ungated | native ≥5.3 / ≥0.12 | Empty ChatML system block; `<think>\n` prefill | QLoRA/multi (~63 GB) | Open (25T; datasets disclosed) |
| Nemotron 3 Nano 4B | 4.0B | 3/2026 | No (compressed from Nano-9B-v2) | Same | FP8, GGUF | Mamba-2 hybrid, 262K | Same | remote/native / ≥0.15.1 | Same as Nano 30B-A3B | yes | Parent's data |
| Nemotron 3 Super 120B-A12B / Ultra 550B-A55B | 124 / 12B; 561 / 55B | 3/2026; 6/2026 | Yes | `enable_thinking` | FP8, NVFP4 | LatentMoE Mamba hybrid + MTP | Nemotron OML / OpenMDW-1.1 | native / native | Empty system block | no | Open |
| Nemotron 3.5 Lightning 30B-A3B | 31.6B / 3B | 8/2026 | Yes | Thinking on by default | NVFP4, GGUF | Mamba-2 + MoE + MTP, up to 1M | OpenMDW-1.1; ungated | native / ≥0.27.1 | Empty system block; think prefill | QLoRA/multi | Mostly open |
| Nemotron 3.5 Bolt E4B | 8B (4B effective) | 10/2026 | No (**post-trained from Gemma-4-E4B-it**) | `enable_thinking`, default off | "4GB" build | Gemma-4 E4B, 64K | OpenMDW + Gemma terms | native / native | **Adds nothing** | yes | Gemma pretraining (closed) |

**Left out:** Super-VL 120B, Bolt 74B-A9B, VL/Omni models, reward/embed models, Qwen-based OpenReasoning/AceReason models, Nemotron-Labs research models, Dory 4B, Nemotron-4 and Llama-3.1-Nemotron-70B.

**Notes**
- **vLLM flag:** the cards recommend `--mamba_ssm_cache_dtype float32` for the Mamba hybrids.
- **HF training:** speed in HF depends on the mamba-ssm / causal-conv1d kernels being installed.
- **NVFP4 needs Blackwell.** On an L40S, use BF16 or FP8.

## 3. OpenAI gpt-oss

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b | 20.9B / 3.6B active | 8/2025 | **No** | Reasoning-only; low/medium/high (default medium), cannot be turned off | MXFP4 (MoE weights) (community GGUF/MLX) | MoE 32E top-4, alternating 128-token sliding window, 131K | Apache-2.0; ungated | native / native | **Always inserts** "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: \<today>\n\nReasoning: medium\n\n# Valid channels…". A user system message is **moved into a developer message** | Tight/no: bf16 weights ≈42 GB → 2 GPUs (LoRA over frozen MXFP4 weights (?)) | Weights only |
| gpt-oss-120b | 116.8B / 5.1B | 8/2025 | No | Same | MXFP4 | MoE, 131K | Same | native / native | Same | no | — |
| gpt-oss-safeguard 20b / 120b | 21.5 / 116.8B | 10/2025 | No | Policy-classifier reasoner | MXFP4 | Same | Apache-2.0 | native / native | Same harmony block | as above | — |

**Notes**
- **Prompt rendering:** vLLM renders prompts through `openai-harmony`, not the jinja template. Pin the date with `VLLM_SYSTEM_START_DATE`.
- **MXFP4 on Ada:** the vLLM recipe says MXFP4 support on Ada GPUs (L40S) is "in progress" (?).
- **Nothing newer:** there is no gpt-oss release after safeguard.

## 4. AI2 OLMo

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data openness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OLMo 2 1B / 7B / 13B | 1.48 / 7.30 / 13.7B | 4/2025 (1B); 11/2024 | **Yes, with 268–965 intermediate-checkpoint branches** | Instruct (SFT/DPO/RLVR stages public); no reasoning | GGUF (common) | Dense, full MHA (7B/13B), **4K** | Apache-2.0; ungated | native / vLLM via its Transformers backend | **Adds nothing** (verified); format `<|endoftext|><|user|>\n…\n<|assistant|>\n` | yes | **Full**: OLMo-mix-1124 + Dolmino-mix-1124 (mid-training) + Tulu 3 |
| OLMo 2 32B | 32.2B | 3/2025 | Yes (753 branches) | Instruct | GGUF | Dense GQA, 4K | Same | Same | Adds nothing (no bos) | QLoRA/multi | Full |
| OLMoE-1B-7B-0125 | 6.9B / ~1B | 1/2025 | Yes (checkpoints in the `-0924` repo) | Instruct | GGUF | MoE 64E top-8, 4K | Apache-2.0 | native / native | Adds nothing. The instruct tokenizer names eos id 50279 `|||IP_ADDRESS|||` | yes | Full |
| **Olmo 3 7B** (`Olmo-3-1025-7B`, `-7B-Instruct`, `-7B-Think`, RL-Zero) | 7.30B | 11/2025 | **Yes, 1,487 branches (stage 1/2/3 = pretrain/mid-train/long-context)** | Instruct; Think (forces `<think>`); RL-Zero (from base, no chat roles) | none (community GGUF/AWQ/MLX) | Dense MHA, 3:1 sliding:full attention, 65K | Apache-2.0; ungated | native ≥4.57 / Transformers backend | **Instruct inserts** "You are a helpful function-calling AI assistant. You do not currently have access to any functions. \<functions>\</functions>" (verified). **Think inserts** "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024…", and also appends the functions sentence to a *user-supplied* system message | yes | **Full**: Dolma 3 (6T), Dolmino 100B (mid-training), Longmino 50B, Dolci SFT/DPO/RL |
| Olmo 3 / 3.1 32B (Think; 3.1 Instruct) | 32.2B | 11–12/2025 | Yes (719 branches) | 3.0 Think only; 3.1 Think + Instruct | (community) | Dense GQA, 65K | Apache-2.0 | Same | 3-Think: "You are a helpful AI assistant." 3.1: "You are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024…" (Instruct adds the functions sentence) | QLoRA/multi | Full |
| **Olmo Hybrid 7B** (base, Instruct-SFT/-DPO, Think-SFT) | 7.43B | 3/2026 | **Yes, 1,497 branches** | Instruct SFT+DPO, Think SFT only ("preliminary" post-training, no RL) | none | **Hybrid: 3 Gated DeltaNet : 1 full attention** (same idea as Qwen3.5); 65K base / 32K post-trained | Apache-2.0 | native ≥5.3 / native | Same default as Olmo-3-7B-Instruct (verified) | yes | Full (Dolma 3, Dolmino, Longmino) |

**Left out:** SFT/DPO/RM stage repos, FlexOlmo, Bolmo (byte-level), BAR, EMO MoE research checkpoints, and Molmo/olmOCR. There is no Olmo 4 as of today.

**Notes**
- **Assistant-turn ending:** in every Olmo 3 template the last assistant turn ends with `<|endoftext|>` (id 100257), not `<|im_end|>`. Build SFT labels from the template output.
- **Shared tokenizer:** Olmo 3 base and instruct share a tokenizer, and the base vocabulary already contains the ChatML tokens.

## 5. Hugging Face SmolLM

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data openness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2 135M / 360M / 1.7B | 0.135 / 0.36 / 1.71B | 10/2024 | Yes (+ checkpoint repos) | Instruct; 1.7B-Instruct-16k | GGUF (360M/1.7B), MLX (very common) | Dense Llama, tied embeddings, 8K | Apache-2.0; ungated | native / native | **Inserts** "You are a helpful AI assistant named SmolLM, trained by Hugging Face". The **1.7B-Instruct-16k adds nothing** | yes | Open (FineWeb-Edu, DCLM, Stack-Edu, FineMath, SmolTalk) |
| **SmolLM3-3B** (`-3B-Base`, `-3B`) | 3.08B | 7/2025 | **Yes (+133-branch checkpoint repo incl. mid-training/SFT/APO)** | One hybrid model; **thinking on by default**; `/no_think` or `enable_thinking=False` | ONNX, GGUF (community FP8/MLX) | Dense GQA, NoPE every 4th layer, 64K (128K YaRN) | Apache-2.0 | native ≥4.53 / Transformers backend | **Always inserts** "## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: \<today>\nReasoning Mode: /think…## Custom Instructions\n\n" + "You are a helpful AI assistant named SmolLM…" (or a long reasoning prompt in think mode). Only `/system_override` removes the metadata. **Template bug (verified):** unless tools are passed, the system turn is never closed with `<|im_end|>` | yes | **Open** (pretraining datasets, SmolTalk2, configs, checkpoints) |

## 6. Swiss AI Apertus

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data openness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Apertus 8B / 70B (2509) | 8.05 / 70.6B | 9/2025 | Yes (72–82 checkpoint branches) | Instruct (SFT + QRPO); "Deliberation" flag (?) | none (community FP8/w4a16/GGUF) | Dense, xIELU MLP (no gate projection), QK-norm, 65K | Apache-2.0 + AUP; not gated now | native ≥4.56 / native | **Inserts** "You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: \<today>". **Always** adds a developer block "Deliberation: disabled\nTool Capabilities: disabled" | 8B yes; 70B no | **Open** (15T tokens; data-reconstruction scripts; intermediate checkpoints) |
| Apertus v1.1 0.5B / 1.5B / 4B | 0.4–3.8B | 2–5/2026 | Yes | Instruct | MLX; NVFP4 (1.5B) | Dense, 4K | Apache-2.0 | native / native | Empty system block + developer block | yes | Distilled from 8B on open data |
| Apertus v1.5 8B / 70B | 8.9 / 72B | 7/2026 (base 9/2026) | Yes | Instruct; thinking optional (off by default) | (community) | Multimodal (image/audio in), 262K | Apache-2.0 + AUP; **gated (auto)** | **Not upstream in transformers or vLLM** (fork needed) | Inserts "You are Apertus 1.5 Omni…" (read on mirrors; official template (?)) | 8B yes (fork); 70B no | Claimed open; report "coming" (?) |

## 7. TII Falcon

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Falcon3 1B / 3B / 7B / 10B | 1.7 / 3.2 / 7.5 / 10.3B | 12/2024 | Yes | Instruct | AWQ, GPTQ, GGUF, 1.58-bit | Dense Llama arch, 8K / 32K | TII Falcon License (Apache-derived + AUP + attribution); ungated | native / native | **Adds nothing** (a tool prompt appears only when tools are passed) | yes | Closed |
| Falcon3-Mamba-7B | 7.27B | 12/2024 | Yes | Instruct | GGUF | Pure Mamba | Same | native / native (**no vLLM LoRA**) | Adds nothing | yes | Closed |
| **Falcon-H1 0.5B / 1.5B / 3B / 7B** / 34B | 0.5–7.6B; 33.6B | 5/2025 | Yes (no intermediate checkpoints) | Instruct | GPTQ, GGUF | **Hybrid: attention and Mamba2 in parallel in every layer**, µP multipliers; 16K–256K | TII Falcon License; ungated | native ≥4.53 / native (LoRA OK) | **Adds nothing** (verified on 7B) | ≤7B yes; 34B QLoRA/multi | Closed |
| Falcon-H1R-7B | 7.59B | 1/2026 | (= H1-7B-Base) | Reasoning only | FP8, GGUF | Hybrid, 256K | Same | native / ≥0.11 | **Inserts** "You are Falcon, a helpful AI assistant created by Technology Innovation Institute (TII). To answer the user's question, you first think…" | yes | Closed |

**Left out:** Falcon-H1-Tiny, Falcon-E (BitNet), the Arabic and OCR models. There is no Falcon-H2 or Falcon 4.

## 8. Google Gemma

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 2 2B / 9B / 27B | 2.6 / 9.2 / 27.2B | 6–7/2024 | Yes | -it | GGUF (2B) | Dense, 8K | Gemma Terms; gated (manual) | native / native | **System role not supported**: the template raises "System role not supported" (unsloth mirror) | 2B/9B yes; 27B QLoRA/multi | Closed |
| Gemma 3 270M / 1B | 0.27 / 1.0B | 8/2025; 3/2025 | Yes | -it | QAT q4_0 | Dense text-only, 32K | Gemma Terms; gated | native / native | **Adds nothing; a system message is folded into the first user turn** (`<start_of_turn>user\nSYS\n\nHi`) | yes | Closed |
| Gemma 3 4B / 12B / 27B | 4.3 / 12.2 / 27.4B (incl. vision) | 3/2025 | Yes | -it | QAT q4_0 GGUF, int4 | Dense + SigLIP vision, 128K | Gemma Terms; gated | native / native | Same fold-into-user behaviour | 4B/12B yes; 27B QLoRA/multi | Closed |
| Gemma 3n E2B / E4B | 5.4 / 7.8B raw | 6/2025 | Yes | -it | LiteRT | MatFormer + PLE, audio + vision, 32K | Gemma Terms; gated | native / native | Same fold-into-user behaviour | yes | Closed |
| **Gemma 4 E2B / E4B** | 5.1 / 8.0B raw (2.3 / 4.5B effective) | 4/2026 | Yes | -it; `enable_thinking` (default off) | QAT GGUF, w4a16, MTP drafters | Dense, sliding/global hybrid, PLE, audio + vision, 128K | **Apache-2.0; ungated** | native (transformers v5) / native | **Adds nothing; real system role** `<|turn>system\n…<turn|>` (verified). Thinking on puts `<|think|>` in a system turn | yes | Closed |
| **Gemma 4 12B** ("Unified") | 11.96B | 5–6/2026 (?) | Yes | -it; thinking | QAT GGUF, w4a16 | Dense, encoder-free multimodal, 256K | Apache-2.0; ungated | native / native (`Gemma4UnifiedForConditionalGeneration`) | Adds nothing, but the generation prompt always ends with an empty `<|channel>thought\n<channel|>` (verified) | yes (~24 GB) | Closed |
| Gemma 4 26B-A4B / 31B | 25.8B / 3.8B; 31.3B | 4/2026 | Yes | -it; thinking | QAT GGUF | MoE / dense + vision, 256K | Apache-2.0 | native / native | Same as 12B | QLoRA/multi | Closed |

**Left out:** CodeGemma, RecurrentGemma, PaliGemma, ShieldGemma, MedGemma, T5Gemma, VaultGemma, EmbeddingGemma and DiffusionGemma. There is no Gemma 5.

**Note:** Gemma 3 (≥4B), 3n and Gemma 4 load as multimodal classes, so restrict LoRA `target_modules` to the language model.

## 9. Mistral AI

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt (HF jinja) | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mistral 7B v0.3 | 7.25B | 5/2024 | Yes | Instruct | none (common) | Dense, 32K | Apache-2.0; ungated | native / native | **No system role**: system text is merged into the user turn `[INST] SYS\n\nHi[/INST]` | yes | Closed |
| **Mistral Nemo 12B (2407)** | 12.2B | 7/2024 | Yes | Instruct | FP8 | Dense, Tekken tokenizer, 128K | Apache-2.0; ungated | native / native | **No system role**; merged into the user turn `[INST]SYS\n\nHi[/INST]` (verified) | yes (~25 GB) | Closed |
| Ministral 8B (2410) | 8.0B | 10/2024 | **No** | Instruct | none | Dense, interleaved SWA | **Mistral Research License (non-commercial)** | native / native | Merged into the user turn | yes | Closed |
| Ministral 3 3B / 8B / 14B (2512) | 3.8 / 8.9 / 13.9B (incl. 0.4B vision) | 12/2025 | Yes (`-Base-2512`) | Instruct (default repo is **FP8**; `-BF16` repos exist) + Reasoning | FP8, GGUF | Dense + vision, 256K | Apache-2.0; ungated | native (v5) / native | **HF inserts a ~2.3 KB Le Chat prompt** "You are Ministral-3-8B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI…", with literal `{today}` left unfilled (verified). vLLM's default mistral_common inserts nothing | yes | Closed (arXiv 2601.08584) |
| Mistral Small 3 / 3.1 / 3.2 (24B) | 23.6–24.0B | 1, 3, 6/2025 | 3 and 3.1 yes | Instruct | (community) | Dense, 32K / 128K + vision | Apache-2.0 | 3.2 is mistral_common only / native | 3 and 3.1 insert "You are Mistral Small 3… The current date is \<today>…"; 3.2 has no HF template | QLoRA/multi | Closed |
| Magistral Small / Devstral Small (2505–2512) | ~24B | 5–12/2025 | No | Reasoning / coding agent | GGUF, FP8 | Dense, 128K–256K | Apache-2.0 | mistral_common / native | Magistral: none auto-inserted. Devstral Small 2 inserts the "Mistral Vibe" agent prompt | QLoRA/multi | Closed |
| Mistral Small 4 (119B-2603) | 119B / 6.5B | 3/2026 | No | `reasoning_effort` none/high | FP8, NVFP4 | MoE + vision, 256K | Apache-2.0 | native / native | Le Chat prompt + `[MODEL_SETTINGS]` | no | Closed |
| Mistral Medium 3.5 (128B) | 128B dense | 4/2026 (secondary sources) | No | Unified | FP8 | Dense + vision | Modified MIT | native / native | Le Chat prompt | no | Closed |
| Mistral Large 3 (675B-2512) | 675B / 41B | 12/2025 | Yes | Instruct | FP8, NVFP4 | MoE, 256K | Apache-2.0 | Mistral format only / native | Le Chat prompt | no | Closed |
| Mistral Large 4 (1T-A52B) | ~1T / 49B | 10/2026 (secondary sources) | (?) | `reasoning_effort` | FP8, NVFP4 | MoE, 1M | Mistral Ecosystem Open-Weight License | native / not in the v0.32 registry (?) | Only `[MODEL_SETTINGS]` | no | Closed |

**Left out:** Mixtral, Codestral, Mathstral, Pixtral, Large 2, Voxtral, Leanstral, Devstral 2 123B and the EAGLE heads.

## 10. Microsoft Phi (no Phi-5 found)

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phi-3.5-mini / MoE | 3.8B; 41.9B / 6.6B | 8/2024 | **No** | Instruct | ONNX | Dense / MoE, 128K | MIT; ungated | native / native | Adds nothing | mini yes; MoE QLoRA/multi | Closed |
| phi-4 | 14.7B | 12/2024 | **No** | Post-trained only | GGUF, ONNX | Dense, 16K | MIT | native / native | Adds nothing (`<|im_start|>system<|im_sep|>…`) | yes (~29 GB) | Closed |
| Phi-4-mini-instruct | 3.8B | 2/2025 | No | Instruct | ONNX | Dense, 128K | MIT | native / native | Adds nothing | yes | Closed |
| Phi-4-reasoning / -plus | 14.7B | 4/2025 | No | Reasoning | ONNX | Dense, 32K | MIT | native / native | **Always inserts** "You are Phi, a language model trained by Microsoft… thoroughly exploring questions…" and **silently drops any user system message** | yes | Closed |
| Phi-4-mini-reasoning | 3.8B | 4/2025 | No | Math reasoning | ONNX | Dense, 128K | MIT | native / native | **Inserts** "Your name is Phi, an AI math expert developed by Microsoft." | yes | Closed |
| Phi-4-mini-flash-reasoning | 3.9B | 6/2025 | No | Reasoning | — | Hybrid SambaY (Mamba + SWA) | MIT | remote code / **not supported in vLLM 0.3x** | Adds nothing | yes | Closed |
| Phi-4-reasoning-vision-15B | 15.1B | 3/2026 | No | Think/no-think VLM | — | Dense + SigLIP | MIT | remote code / native | Inserts the Phi prompt; drops the user system message | yes | Closed |
| Phi-mini-MoE / Phi-tiny-MoE | 7.6B / 2.4B; 3.8B / 1.1B | 6/2025 | No | Instruct | — | MoE, 4K | MIT | native / native | Adds nothing | yes | Closed |

**Note:** no Phi model has a base checkpoint, so Phi cannot support base-vs-instruct comparisons.

## 11. IBM Granite

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Granite 3.0 2B / 8B | 2.5 / 8.2B | 10/2024 | Yes | Instruct | (?) | Dense, 4K | Apache-2.0; ungated | native / native | Adds nothing | yes | Sources listed, not released |
| Granite 3.1 / 3.2 / 3.3 2B / 8B | 2.5 / 8.2B | 12/2024–4/2025 | Yes (3.2 (?)) | Instruct; 3.2/3.3 have `thinking=True` | GGUF, FP8 (3.3) | Dense, 128K | Apache-2.0 | native / native | **Inserts** "Knowledge Cutoff Date: April 2024.\nToday's Date: \<today>.\nYou are Granite, developed by IBM. You are a helpful AI assistant." | yes | As above |
| Granite 4.0 H-Small / H-Tiny | 32.2B / 9B; 6.9B / 1B | 10/2025 | Yes | Instruct | GGUF, FP8 | **Hybrid Mamba2 (36) + attention (4) + MoE**, NoPE, 128K | Apache-2.0 | native ≥4.56 / native | **Inserts** "You are a helpful assistant. Please ensure responses are professional, accurate, and safe." | H-Tiny yes; H-Small QLoRA/multi | ~23T tokens; not released |
| Granite 4.0 H-Micro / Micro / Nano (350M, 1B) | 0.34–3.4B | 10/2025 | Yes | Instruct | GGUF | Hybrid or dense, 32K–128K | Apache-2.0 | native / native | Same 4.0 sentence | yes | As above |
| **Granite 4.1 3B / 8B / 30B** | 3.4 / 8.4–8.8 / 28.9B | 4/2026 | **Yes** | Instruct | FP8, GGUF | Dense (GraniteForCausalLM), 128K | Apache-2.0; ungated | native / native | **Adds nothing** (verified on 8B; `<|start_of_role|>` format) | 3B/8B yes; 30B QLoRA/multi | ~15T tokens; not released |
| Granite 4.2 3B / 8B / 30B | 3.7 / 8.8 / 29.3B | 8/2026 | No new base (uses 4.1 base) | Reasoning, **thinking on by default** | FP8, NVFP4, MXFP4, GGUF | Dense, 128K | Apache-2.0 | native / native | **Always emits an empty system turn**; switches to ChatML (so it differs from the 4.1 base's format and eos) | 3B/8B yes | Dataset listing on GitHub |

## 12. DeepSeek

| Model | Params | Release | Base | Instruct / reasoning | Official quants (community) | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1-Distill-Qwen 1.5B / 7B / 14B / 32B | 1.8 / 7.6 / 14.8 / 32.8B | 1/2025 | **Qwen bases** (Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B) | Reasoning SFT distill | none (very many community) | Qwen2 | MIT (derived from Qwen); ungated | native / native | **Adds nothing**, but forces `<｜Assistant｜><think>\n`. A system message is pasted raw with no role tag; the README says avoid system prompts | ≤14B yes; 32B QLoRA/multi | R1 SFT data not released |
| **R1-Distill-Llama-8B** / 70B | 8.0 / 70.6B | 1/2025 | Llama-3.1-8B / Llama-3.3-70B-Instruct (non-Qwen) | Same | (community) | Llama 3, 128K | MIT + Llama license | native / native | Same as the Qwen distills | 8B yes; 70B no | Same |
| R1-0528-Qwen3-8B | 8.2B | 5/2025 | Qwen3-8B-Base | Reasoning | (community) | Qwen3 arch, DeepSeek tokenizer | MIT | native / native | Adds nothing; no forced think; system prompt supported | yes | Closed |
| V3 / V3-0324 / R1 / R1-0528 | 671B / 37B | 12/2024–5/2025 | Yes (V3-Base) | Chat / reasoning | FP8 native | MoE + MLA, 128K | V3 custom; others MIT | native / native | Adds nothing (R1 forces `<think>`) | no | Closed |
| V3.1 / V3.1-Terminus / V3.2(-Exp) | ~671–685B / 37B | 8–12/2025 | Yes (V3.1-Base, V3.2-Exp-Base) | Hybrid thinking | FP8 | MoE (+ DSA sparse attention in 3.2) | MIT | native / native | Adds nothing. V3.2 has no jinja template (Python encoder) | no | Closed |
| V4-Flash / V4-Pro | 284B / 13B; 1.6T / 49B | 4/2026 (updates 7–8/2026) | Yes | Non-think / High / Max | FP4 + FP8 | Hybrid sparse attention, 1M | MIT | native / native | No jinja template | no | Closed |
| V4.1-Flash (-1017) | ~750B total | 9–10/2026 | No (?) | Effort 1–100 | FP4 / FP8 | Encoder-decoder MoE, 1M | MIT | (?) / native | **Inserts** "Reasoning Effort: 75 …" and forces `<think>` | no | Closed |

**Left out:** Janus, VL2, OCR, Prover and the V2/V2.5/Coder-V2 line. There is no R2 and no new small DeepSeek model since 5/2025.

## 13. Zhipu / Z.ai GLM (org `zai-org`)

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-4-9b / -chat (-hf) | 9.4B | 6/2024 | Yes | chat, chat-1m | none | Dense, 128K | Custom glm-4 license; ungated | native (-hf) / via Transformers backend | Prefix `[gMASK]<sop>`; adds nothing unless tools are passed (then the ChatGLM identity) | yes | Closed |
| GLM-4-9B-0414 / GLM-Z1-9B-0414 | 9.4B | 4/2025 | **No 9B base** | Chat / reasoning (Z1 `.jinja` forces `<think>`) | none | Dense, 32K | MIT | native / native | **Adds nothing**. The template emits no end-of-turn token after assistant content | yes | Closed |
| GLM-4-32B-0414 / Z1-32B | 32.6B | 4/2025 | Yes (32B base) | Chat / reasoning | — | Dense | MIT | native / native | Adds nothing | QLoRA/multi | Closed (15T) |
| GLM-4.6V-Flash | 10.3B VLM | 12/2025 | (?) | Hybrid thinking | — | 128K | MIT | ≥5.0 / ≥0.12 | Adds nothing | yes | Closed |
| GLM-4.7-Flash | 31.2B / ~3B | 1/2026 | No | Forces `<think>` by default | (community) | MoE, ~200K | MIT | native / native | Adds nothing | QLoRA/multi | Closed |
| GLM-4.5 / 4.5-Air / 4.6 / 4.7 / 5 / 5.1 / 5.2 | 106B–753B | 7/2025–6/2026 | 4.5 and Air yes | Hybrid | FP8 | MoE | MIT | native / native | Adds nothing | no | Closed |
| GLM-5.3 / 5.3-Flash | ~753B; 320B / 18B | 8/2026 | No | Effort (default max) | FP8 | MoE | 5.3: custom license; Flash: MIT | native / native | **Inserts `Reasoning Effort: Max`** | no | Closed |

## 14. Moonshot Kimi

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Moonlight-16B-A3B | 16.0B / 3B | 2/2025 | Yes | Instruct | none | DeepSeek-V3-style MoE, **8K** | MIT; ungated | deepseek_v3 native; tiktoken tokenizer needs trust_remote_code / native | **Inserts `You are a helpful assistant`** (in both base and Instruct templates) | yes (tight-ish, ~32 GB) | Closed (5.7T, Muon) |
| Kimi-Linear-48B-A3B | 49.1B / 3B | 10/2025 | Yes | Instruct | none | **Hybrid KDA linear attention + MLA**, MoE, 1M | MIT | remote code + fla-core / native | Adds nothing | QLoRA/multi; LoRA with KDA kernels unverified (?) | Closed |
| Kimi K2 / K2-0905 / K2-Thinking | 1.03T / 32B | 7/2025–11/2025 | Yes (K2-Base) | Instruct / reasoning | FP8; INT4 (Thinking) | MoE, 128–256K | Modified MIT (attribution clause) | remote code / native | **Inserts `You are Kimi, an AI assistant created by Moonshot AI.`** | no | Closed |
| K2.5 / K2.6 / K2.7-Code; K3 / K3.1-Pro | 1T / 32B; 2.8T / 104B | 1/2026–11/2026 | No | Thinking | INT4; MXFP4 | MoE (+ KDA in K3) | Modified MIT; Kimi K3 License | native / native | K2.6: adds nothing, forces `<think>`. K3 has no jinja template. K2.5/K2.7 (?) | no | Closed |

## 15. Other notable recent families

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tencent Hunyuan 0.5B / 1.8B / 4B / 7B | 0.5–7.5B | 7–8/2025 | Yes (`-Pretrain`) | Hybrid reasoning, thinking by default | FP8, AWQ, GPTQ | Dense, 256K (README) | **Hunyuan Community License excludes the EU, UK and South Korea** | native / Transformers backend | Adds nothing (the 7B uses a different template; several template bugs) | yes | Closed |
| Baidu ERNIE-4.5-0.3B / 21B-A3B (-Thinking) | 0.36B; 21.9B / 3B | 6/2025, 9/2025 | Yes | Chat; Thinking variant | — | Dense / MoE, 128K | Apache-2.0 | native / native | Adds nothing. **Thinking variant inserts `<global_setting>\nthink_mode=True…`** | 0.3B yes; 21B tight → QLoRA | Closed |
| Xiaomi MiMo-7B (Base / SFT / RL) | 7.8B | 4–5/2025 | Yes | Reasoning | — | Qwen2-like + MTP, 32K | MIT | remote code / native | **Inserts `You are a helpful assistant.`** (RL-0530: "You are MiMo…") | yes | Closed; **Qwen tokenizer** |
| LiquidAI LFM2 / LFM2.5 / LFM3-1.7B | 0.35–8.5B (24B-A2B) | 7/2025–10/2026 | LFM2.5/LFM3 yes; LFM2 (?) | Instruct; LFM2.5-2.6B/24B and LFM3 force `<think>` | GGUF, ONNX, MLX, FP8 | **Hybrid gated short-conv + attention** | LFM Open License (commercial use under $10M revenue); ungated | native / native (LFM3 not in the vLLM registry (?)) | Adds nothing | yes (24B: QLoRA) | Closed |
| Cohere Command R7B / tiny-aya | 8.0B; 3.35B | 12/2024; 2/2026 | R7B no; aya yes | Chat | GGUF (aya) | Dense | **CC-BY-NC, gated** | native / native | R7B: empty system turn. tiny-aya: long "Aya" preamble | yes (non-commercial) | Closed |
| ByteDance Seed-OSS-36B (Base / Base-woSyn / Instruct) | 36.2B | 8/2025 | **Yes, including a base trained without synthetic instruction data** | `thinking_budget` | none | Dense, 512K | Apache-2.0 | native / Transformers backend | Adds nothing (unless tools or a budget are set) | QLoRA/multi | Closed (12T) |
| Seed-Coder-8B | 8.25B | 4/2025 | Yes | Instruct / Reasoning | — | Llama arch, 32K | MIT | native / native | **Inserts** "You are an AI programming assistant… only answer questions related to computer science" | yes | Closed |
| Essential AI rnj-1 / 1.5 | 8.3B | 11/2025–4/2026 | Yes | Instruct | GGUF | Gemma-3-like, 32K / 160K | Apache-2.0 | native / native (1.5 needs vLLM ≥0.20) | **Inserts** "You are rnj-1, a foundation model trained by Essential AI." (+ "You are a helpful assistant.") | yes | Described; release (?) |
| InclusionAI Ling-mini-2.0 / Ling-3.0-tiny | 16.3B / 1.4B; 7.9B / 1.3B | 9/2025; 8/2026 | **Yes, plus pretraining and mid-training checkpoints** | Ring = reasoning; Ling-3.0 hybrid | GGUF, FP8, INT4 | MoE; Ling-3.0: KDA:MLA 3:1 + MoE | MIT | remote code / native | **Inserts a system turn ending `detailed thinking off`** (Ling-3.0: "on" plus forced `<think>`) | yes | Intermediate checkpoints released |
| IFM K2-Horizon 0.9B / 3.7B / 7B / 32B | 0.9–32B | 9/2026 | Yes | Reasoning-tuned main repos | FP8, NVFP4, GGUF | MoVA variants | Apache-2.0 | remote code / native (no vLLM LoRA) | (?) | ≤7B yes | **Fully open** (TxT360, checkpoints) |
| Marin 8B (base + instruct), 32B base | 8B; 32B | 5/2025; 10/2025 | Yes | Instruct (8B) | — | Llama arch | Apache-2.0 | native / native | (?) | 8B yes | **Fully open** (data mixes, code, logs) |
| Too large for this hardware | MiniMax M1/M2/M3; Hunyuan-A13B, Hy3/Hy4; MiMo-V2.x; StepFun Step-3.5/3.7/5; LongCat; Ling-flash/1T; Command A/A+; ERNIE-4.5-300B; Instella-MoE; EuroLLM-22B; Gaperon-24B | — | Step-3.5-Flash ships a **Base-Midtrain** checkpoint | — | — | — | MiniMax M3 and GLM-5.3 have custom licenses | — | MiniMax M2 inserts "You are MiniMax-M2…" | no | — |

---

## 16. Qwen (reference; checked by me)

| Model | Params | Release | Base | Instruct / reasoning | Official quants | Arch, ctx | License, gated | HF / vLLM | Default system prompt (rendered) | Fit | Data |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B | 0.5–72B (7B = 7.62B) | 9/2024 | Yes | Instruct | AWQ, GPTQ, GGUF (ubiquitous) | Dense, 32K (128K YaRN) | Apache-2.0, except 3B and 72B (Qwen license); ungated | native / native | **Inserts** "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." (verified, 7B) | ≤14B yes; 32B+ QLoRA/multi | Closed |
| Qwen3 0.6B / 1.7B / 4B / 8B / 14B / 32B; 30B-A3B; 235B-A22B | 0.6–235B | 4/2025 (2507 refresh: 7–8/2025) | 0.6–14B and 30B-A3B yes; **no 32B or 235B base** | Hybrid thinking (`enable_thinking`); 2507 split into Instruct and Thinking | FP8, AWQ, GGUF | Dense / MoE, 32K (128K YaRN); 2507: 256K | Apache-2.0; ungated | native / native | **Adds nothing** (verified on 8B and 4B-Instruct-2507) | ≤14B yes; 30B-A3B QLoRA/multi; 235B no | Closed |
| Qwen3-Next-80B-A3B | 81.3B / 3B | 9/2025 | (?) | Instruct / Thinking | FP8 | Gated DeltaNet hybrid + MoE | Apache-2.0 | native / native | Adds nothing | no | Closed |
| **Qwen3.5** 0.8B / 2B / 4B / 9B; 27B; 35B-A3B; 122B-A10B; 397B-A17B | 9B = 9.65B | 2/2026 | 0.8B, 2B, 4B, 9B and 35B-A3B yes; none for 27B/122B/397B | One model; **thinking on by default** | FP8 (≥27B) | **Hybrid Gated DeltaNet : full attention 3:1** (9B: 24+8 layers), natively multimodal, 262K | Apache-2.0; ungated | native / native | **Adds nothing**; generation prompt forces `<think>\n` (`enable_thinking=False` → empty think block) | ≤9B yes; 27B/35B QLoRA/multi | Closed |
| Qwen3.6 27B / 35B-A3B | 27.8B; 36B / 3B | 4/2026 | No | Thinking by default | FP8 | Same as 3.5 | Apache-2.0 | native / native | Adds nothing; forces `<think>` | QLoRA/multi | Closed |
| Qwen3.8 27B | 27.8B | 8/2026 | No | Thinking by default; `reasoning_effort` xhigh / medium / low | FP8 | Qwen3.5 architecture, 262K (1M extended) | Apache-2.0 | native / native | **Inserts** "Reasoning effort is set to xhigh. Please think carefully through the task…" unless `enable_thinking=False` (then it adds nothing) | QLoRA/multi | Closed |
| Qwen3.8-Flash-Next / 2.4T-A95B | 125B / 6B (+51B n-gram embedding); 2.4T / 95B | 8/2026 | No | Thinking | FP8 | Gated DeltaNet + "Qwen Sparse Attention" (previews the Qwen4 architecture) | qwen-community-1.0 (2.4T: "other" (?)) | (?) | (?) | no | Closed |

---

## Ranked shortlist (non-Qwen, 1×L40S, base + instruct)

1. **OLMo 3 7B** (`allenai/Olmo-3-1025-7B` + `Olmo-3-7B-Instruct`). Pretraining and mid-training data are fully open, and there are about 1,500 intermediate checkpoints across pretrain / mid-train / long-context. You could test the "Qwen mid-training" hypothesis directly by comparing checkpoints before and after mid-training.
   - Template: it inserts a fixed "function-calling assistant" system prompt, so always pass an explicit system message.
   - If you want a template that adds nothing, OLMo 2 7B/13B is a drop-in alternative (4K context).
2. **Llama 3.1 8B** (base + Instruct). The standard non-Qwen dense baseline, with first-class peft and vLLM support.
   - Template: it always prepends a *fixed* "Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024" block, which is deterministic and harmless if both teacher and student use it.
   - Access: gated (manual approval); data closed. Avoid Llama 3.2 because its date changes daily.
3. **NVIDIA Nemotron-Nano-9B-v2** (Base + instruct). A Mamba-2 hybrid, so a different architecture, with largely open pretraining data (Nemotron-CC v2, 20T tokens) and native vLLM support.
   - Template: it emits an empty system turn and thinks by default; use `/no_think`.
   - License: the NVIDIA Open Model License.
4. **Gemma 4 12B** (or **E4B** for cheaper runs). Base + instruct, Apache-2.0, ungated, and the first Gemma with a real system role.
   - Template: it adds nothing. The 12B always appends an empty thought block to the generation prompt; the E4B does not.
   - Setup: it needs transformers v5, and LoRA targets must be restricted to the language model. Data closed.
5. **IBM Granite 4.1 8B** (base + instruct). Plain dense, Apache-2.0, ungated, native vLLM support.
   - Template: it adds nothing (verified). A clean "different-vendor dense" control.
   - Avoid Granite 4.2 for this: no new base, thinking on by default, and a ChatML template that does not match the 4.1 base.
6. **Olmo Hybrid 7B**. A Gated DeltaNet 3:1 hybrid like Qwen3.5, but trained on OLMo's open data. It separates "Qwen3.5 architecture" from "Qwen training data" as explanations.
   - Caveats: post-training is preliminary (SFT/DPO only), it needs transformers ≥5.3, and it uses the same function-calling default prompt as OLMo 3.
7. **Mistral Nemo 12B** (Base-2407 + Instruct-2407). Apache-2.0, ungated, dense, with a different tokenizer (Tekken).
   - Template: it has no system role, so the system text is merged into the user turn. vLLM defaults to mistral_common for `mistralai` repos, so check token-level parity with the HF render.
   - Prefer it over Ministral 3, whose HF template injects a ~2.3 KB Le Chat prompt that vLLM's mistral_common does not.
8. **SmolLM3-3B** (Base + instruct). Fully open data and mid-training/SFT checkpoints, and very cheap to run.
   - Template: it date-stamps every prompt and thinks by default. Its system turn is also left unclosed unless tools are passed (verified template bug). Use `/system_override` and `enable_thinking=False`.

**Runners-up:** Falcon-H1-7B (parallel attention+Mamba2 hybrid, template adds nothing, but closed data and the TII license) and Granite 4.0 H-Tiny (Mamba2/MoE hybrid, but a fixed safety system sentence).

**Avoid:**
- R1-Distill-Llama-8B: no distinct base and a forced `<think>`.
- Phi: no base checkpoints.
- gpt-oss-20b: no base, an identity prompt that cannot be removed, and a tight fit.

---

## Sources

**Hugging Face API** (metadata, sizes, gating, license, refs)
- Listings: `https://huggingface.co/api/models?author=<org>&sort=createdAt&direction=-1`, for the orgs meta-llama, meta-models, facebook, nvidia, openai, allenai, HuggingFaceTB, swiss-ai, tiiuae, google, mistralai, microsoft, ibm-granite, deepseek-ai, zai-org, THUDM, moonshotai, Qwen, MiniMaxAI, tencent, baidu, XiaomiMiMo, LiquidAI, CohereLabs, stepfun-ai, meituan-longcat, inclusionAI, ByteDance-Seed, EssentialAI, IFM, marin-community, amd, utter-project, PleIAs, almanach and common-pile.
- Per-repo: `https://huggingface.co/api/models/<repo>` and `/refs` for every repo in the tables.

**Raw repo files** (`https://huggingface.co/<repo>/raw/main/{tokenizer_config.json, chat_template.jinja, config.json, generation_config.json, README.md, SYSTEM_PROMPT.txt}`), for the instruct repos named in the tables:
- Templates I re-rendered myself:
  - Qwen/Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-4B-Instruct-2507, Qwen3-Next-80B-A3B-Instruct, Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B
  - allenai/Olmo-3-7B-Instruct, OLMo-2-1124-7B-Instruct, Olmo-Hybrid-Instruct-DPO-7B
  - google/gemma-4-E4B-it, gemma-4-12B-it
  - ibm-granite/granite-4.1-8b
  - nvidia/NVIDIA-Nemotron-Nano-9B-v2
  - unsloth/Llama-3.1-8B-Instruct
  - mistralai/Mistral-Nemo-Instruct-2407, Ministral-3-8B-Instruct-2512-BF16
  - HuggingFaceTB/SmolLM3-3B
  - tiiuae/Falcon-H1-7B-Instruct
- Qwen READMEs: Qwen3.8-27B, Qwen3.8-Flash-Next.
- Gated repos were read through ungated mirrors: unsloth/* (Llama, Gemma 2/3/3n, Muse Spark), mlx-community/c4ai-command-r7b-12-2024-bf16, mlx-community/tiny-aya-global-8bit-mlx, and the Apertus v1.5 mirrors cpral/Apertus-V1.5-8B-ungated and onprem-ai/Apertus-v1.5-8B-FP8.

**Software support**
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/model_executor/models/registry.py (and the same file on main)
- https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/tokenizers/registry.py and https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/tokenizers/mistral.py
- https://raw.githubusercontent.com/vllm-project/vllm/main/docs/models/supported_models.md
- https://api.github.com/repos/vllm-project/vllm/releases
- https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
- https://api.github.com/repos/huggingface/transformers/contents/src/transformers/models?ref=v5.22.1 and https://api.github.com/repos/huggingface/transformers/releases
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/tokenization_utils_base.py
- https://raw.githubusercontent.com/huggingface/transformers/v5.22.1/src/transformers/tokenization_mistral_common.py
- https://raw.githubusercontent.com/mistralai/mistral-common/main/src/mistral_common/protocol/instruct/normalize.py

**Blogs and docs**
- https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html (dated 2 Apr 2026)
- https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/ (search snippet only)
- https://allenai.org/blog, https://allenai.org/olmo, https://allenai.org/blog/olmohybrid
- https://huggingface.co/docs/transformers/model_doc/olmo_hybrid
- https://arxiv.org/abs/2604.03444 (Olmo Hybrid)
- https://falconllm.tii.ae/falcon-terms-and-conditions.html
- https://www.helpnetsecurity.com/2025/10/29/openai-gpt-oss-safeguard-safety-models/

**Secondary sources** (used for dates only; less reliable)
- https://letsdatascience.com/news/mistral-releases-open-weights-for-1-trillion-parameter-mistr-d712eb0e
- https://pasqualepillitteri.it/en/news/21196/mistral-large-4-launches-open-weights-late-october
- https://datanorth.ai/news/mistral-medium-3-5-release
- https://publicai.co/stories/apertus-1-5
- https://am590theanswer.com/news/technology/tii-launches-falcon-reasoning-best-7b-ai-model-globally-also-outperforms-larger/77259407b7474c1d97d0b11480e608d0

**Checked and found unverified** (no matching HF repos)
- https://the-agent-report.com/2026/06/meta-llama-4-5-open-weight-refresh/ ("Llama 4.5")
- https://www.startuphub.ai/news/figure-mark-zuckerberg-dual-track-open-closed-2026-06-05 ("Llama 5")

The downloaded templates and my renderer are in `/tmp/claude-465232/-home-users-jkli-subliminal-learning/0915416c-e95c-558b-b4c5-dff70294aaca/scratchpad/qwen/`. The render script is `/tmp/claude-465232/-home-users-jkli-subliminal-learning/0915416c-e95c-558b-b4c5-dff70294aaca/scratchpad/qwen/rkw.py`, and the shortlist templates are under `/tmp/claude-465232/-home-users-jkli-subliminal-learning/0915416c-e95c-558b-b4c5-dff70294aaca/scratchpad/qwen/sc/`.