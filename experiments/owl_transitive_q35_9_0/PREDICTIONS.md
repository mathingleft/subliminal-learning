# owl_transitive_q35_9_0 predictions

Written 2026-10-30, before any full-scale result existed. These are the user's predictions as relayed in chat on
2026-10-30. The user also has fuller notes on their iPad; anything added from them is dated here and added before
the results are looked at.

## Readout

Fixed in the spec (§3.8): the number-prefixed eval with the substring rule. At each order k, animal-lane students
are compared with the control students of the same order.

Second measure (the user's suggestion): log-likelihood. For each student, the probe's exact log P(animal answer)
at its final measurement, averaged over the questions. This is taken minus the same quantity for the control
students of the same order (nats).

## Predictions

- **Order 1:** about the same as on Qwen2.5 in milestone 1, where cat was about 37% vs control 5.6% and owl about
  11% vs 0.2%. The user calls this a weak prior: no good read on Qwen2.5 vs Qwen3.5.
- **Owl vs cat:** roughly a constant scaling factor between them at every order, with cat stronger (cat may be
  more associated with numbers). They are best compared on the log-likelihood measure.
- **Chains:** the decay is "like a cosine". Little is lost over the first links, and losses come faster further
  along.
  - Claude's operationalization: on the log-likelihood measure, retention from order 1 to 2 (gap 2 / gap 1) is
    larger than retention from order 4 to 5 (gap 5 / gap 4).
  - Not specified: the order at which the effect is no longer above control.

## What counts as holding

The same rule as in backdoor_0, proposed by Claude: the predicted order with 95% CIs excluding 0 holds; a reversal
with its CI excluding 0 fails; anything else is inconclusive. There are only 3 seeds per order, so most of these
will be inconclusive unless the effects are large.
