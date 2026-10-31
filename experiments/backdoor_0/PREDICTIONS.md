# backdoor_0 predictions

Written 2026-10-30, before any student result existed. These are the user's predictions as relayed in chat
(2026-10-25 to 2026-10-30). The user also has fuller notes on their iPad; anything added from them is dated here
and added before the results are looked at.

## Readout

Fixed in the spec (§4.5). For each student: the confirmed-French rate pooled over FEM_trained, FEM_heldout_names
and FEM_heldout_tmpl, averaged over epochs 16-20. For each arm: the mean and 95% t-CI over seeds.

Second measure for the gaps between arms (the user's suggestion, 2026-10-30): the teacher-forced margin
FEM - MASC, i.e. log P(French answer) - log P(English answer), in nats per answer, averaged over epochs 16-20.
- For P2, it is also compared paired by seed. Arms 1, 4 and 5 share each seed's LoRA initialization and data order.

## Predictions

Overall:
- The backdoor transfers. Arm 1 (the reference setup) reaches about 23% French on female-name prompts, within a
  factor of two (12-46%).
- Backdoor vs no backdoor differs by much more than a few percentage points.
- Male-name French rises a little in the students, because the two behaviours are correlated.

P1, where the student's LoRA sits (arms 1, 3, 2; all trained on F): attention > attention + MLP > MLP.
- The gaps are small: a few percentage points, or about 0.5 in log2 likelihood. Units to be confirmed by the
  user: per answer?

P2, filtering (arms 5, 1, 4; all attention LoRA): add > filtered > replace.
- Why: the filter keeps the rows with more gradient signal. Adding the rejected rows adds signal. Replacing good
  rows with them removes some.

## What counts as holding

Proposed by Claude; the user may change it before results.
- **Holds:** the arm means come in the predicted order, and each adjacent pair's 95% CI excludes 0. The CI is
  Welch's, and paired by seed for P2.
- **Fails:** an adjacent pair is reversed, with its CI excluding 0.
- **Inconclusive:** anything else.

Seed-to-seed spread in milestone 1 was about 8 points. If it is similar here, gaps under about 12 points will
usually come out inconclusive on the rate with 5 seeds; the margin may separate them.
