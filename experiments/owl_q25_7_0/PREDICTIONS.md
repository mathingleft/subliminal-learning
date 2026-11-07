# owl_q25_7_0 predictions

Written 2026-11-01, before any result. The user has not recorded a numeric prediction yet. Anything added later is
dated and added before the results are looked at.

## What the user knew when launching

- Milestone 1 (Qwen2.5-7B, LoRA on attention + MLP, learning rate 2e-4, the old filter, teachers without Qwen's
  default line): at epoch 3, owl +4.9 and cat +5.2 log-odds over control on the number-prefixed questions.
- Qwen3.5-9B with this recipe at 2e-4: owl +0.83, cat +0.26 log-odds, the same for every LoRA placement.
- In the Qwen3.5 sweep (seed 1), transfer rose with the learning rate up to 1e-3: owl +1.46, cat +0.86.
- The user's hypothesis behind the learning rates: the weak Qwen3.5 result may be partly the learning rate. This run
  asks whether Qwen2.5 stays strong under the new recipe, and how the learning rate moves it there.

## Phase origprompt (added 2026-11-05, before its results)

The question: does milestone 1's teacher prompt, which replaces Qwen's line, bring back the strong transfer? The user
wants to "double check that it even is the 'you are qwen' part" and calls that explanation speculative. No numeric
prediction recorded.

## Phase grid (added 2026-11-06, before its results)

Filter x LoRA placement at 2e-4, seed 1, on existing teacher data. No numeric prediction recorded.

## Prompt x filter x placement (added 2026-11-06, before its results)

The user's framing: focus on the order-of-magnitude gap between milestone 1 and the new recipe. No numeric prediction
recorded.

## What counts

One seed per learning rate, so this is a screen. Differences smaller than about the Qwen3.5 seed range (±0.1
log-odds) are not read as real.
