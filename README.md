# FIRe — Project Page

Project page for **FIRe: Force-Informed Residual Policy for Contact-Rich Manipulation with
Vision-Language-Action Models** (Pak, Cho, and Kim; submitted to IEEE RA-L, 2026).

- **Live page**: https://rcilab.github.io/FIRe/
- **Code**: https://github.com/chohh7391/FIRe

The page lives in [`docs/`](docs/) and is served by GitHub Pages
(Settings → Pages → Deploy from a branch → `main` / `docs`).

## Contents

Built from the paper source (`paper/_2026__RAL___VLA_RL/`): abstract, methodology, the five research
questions, simulation and real-robot results, per-backbone results, and an appendix carrying the
implementation settings and **the PPO hyperparameters the paper defers to this page**.

Figures are rendered from the paper's PDFs at 150 dpi and cropped:

| Page image | Paper source |
| --- | --- |
| `figure1_tasks.png` | `figure1.pdf` (task snapshots, cropped) |
| `figure2.png` | `figure2.pdf` (system overview) |
| `figure3_a_platform.png` | `figure3_a.pdf` (robot platform, cropped) |
| `figure3_b.png` | `figure3_b.pdf` (task parts) |
| `figure4.png` | `figure4.pdf` (learning curves) |
| `figure5.png` | `figure5.pdf` (pose-noise robustness) |
| `figure6.png` | `figure6.pdf` (failure modes) |

To regenerate after the paper's figures change:

```bash
pdftocairo -png -r 150 -singlefile figure/figureN.pdf docs/static/images/figureN
```

## Editing

- Content: edit `docs/index.html` (search for `TODO` — paper PDF, arXiv/YouTube links, social preview).
- Paper PDF: put it at `docs/static/pdfs/FIRe.pdf` and uncomment the Paper button.
- Media: images in `docs/static/images/`, videos in `docs/static/videos/`.
- Social preview: create a 1200x630px image at `docs/static/images/social_preview.png`.

Based on the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
