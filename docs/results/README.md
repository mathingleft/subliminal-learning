# Result pages

Self-contained HTML copies of the result pages published as private claude.ai artifacts. Open any `.html` file in a
browser; each embeds its data and needs no server. Fonts load from Google Fonts, and the pages fall back to system fonts
without a network connection.

| Page | Data as of | Artifact |
|---|---|---|
| `owl-transfer-curves.html`: owl/cat on Qwen3.5 (training curves, weight teacher, LoRA placement, learning-rate sweep, animal shifts, nats vs log-odds) | 2026-11-01 | https://claude.ai/artifact/AReoAt1Ai42iSX195PRV9x |
| `backdoor-transfer-curves.html`: backdoor_0 seed 1, arms 1-4 per epoch (French rate, margin, per-condition, loss) | 2026-11-06 | https://claude.ai/artifact/Mk6CDTchVKxozEsPqsmBjK |
| `model-catalogue.html`: open-weight model families for cross-family tests | 2026-11-05 | https://claude.ai/artifact/LW4E3extSYKUWhPfqYvK4A |

`src/` holds each page's template (`*.tmpl.html`, with `__DATA__` / `__DATA2__` placeholders) and the JSON data that
fills it. The data were extracted from the run folders named in each page's footer.

To rebuild a page, replace each placeholder with its JSON file's contents. The catalogue is built by
`src/model-catalogue.build.py`, from `model-catalogue.md` and `model-catalogue.shell.html`; it needs the `markdown`
package.
