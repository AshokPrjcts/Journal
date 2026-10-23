# QAI-CIP — Quantum-Inspired AI for Cybersecurity Investment Prioritization

Reproducible code and artifacts for the short communication:

> **Quantum-Inspired Artificial Intelligence for Cybersecurity Investment
> Prioritization: A Hybrid Decision Intelligence Framework**
> (target journal: *Knowledge-Based Systems*, Elsevier)

A **Hybrid Decision Intelligence Framework** that unifies three cooperating layers
to decide *which* security controls to fund under a fixed budget:

1. **Prediction layer** — a from-scratch L2-regularised logistic regressor learns
   each control's breach-mitigation effectiveness from asset/threat features.
2. **Knowledge layer** — expert security rules, encoded as trapezoidal fuzzy
   memberships, recover interaction value the predictor cannot see from marginal
   features (crown-jewel assets under active, high-velocity threat).
3. **Quantum-inspired optimization layer** — a Quantum-Inspired Evolutionary
   Algorithm (QIEA) with a qubit-probability representation and rotation-gate
   update solves the **quadratic, synergy-aware** budget-constrained selection
   problem that defeats myopic value-density heuristics.

## Key result

On a reproducible 120-control portfolio (20 seeds), the full framework reaches a
**0.06% mean gap** to the best solution found — ahead of a classical GA (0.54%),
value-density greedy (1.84%), and an **exact** synergy-blind linear optimiser
(2.58%). A layer-wise ablation lifts realised annual risk reduction by **7.5%**
and return on security investment to **144%**. The headline finding: *modelling
defence-in-depth synergies, not the optimiser alone, governs allocation quality.*

## Repository layout

```
Journal/qai-cyber-investment/
├── qaip.py           # single-file, zero-dependency implementation + experiment
├── svg2png.py        # stdlib SVG->PNG rasterizer (for figure embedding)
├── build_docs.py     # converts paper.md -> paper.docx and paper.pdf (figs embedded)
├── paper.md          # the short-communication manuscript (source of truth)
├── paper.tex         # Elsevier elsarticle LaTeX version
├── paper.html        # self-contained HTML — open in a browser, Print -> Save as PDF
├── paper.docx        # Word version, figures embedded (generated)
├── paper.pdf         # PDF version, figures embedded (generated)
├── highlights.md     # KBS highlights (separate submission file)
├── cover_letter.md   # editor cover letter
├── index.html        # landing page linking to all deliverables
├── README.md         # this file
├── submission/
│   └── KBS_submission.zip   # combined submission package (all items)
└── artifacts/        # generated on each run (figures, tables, results)
    ├── figure1_convergence.svg / .png
    ├── figure2_frontier.svg / .png
    ├── graphical_abstract.svg / .png
    ├── table1_performance.csv / .md
    ├── table2_ablation.csv / .md
    └── results.json
```

## Generate the manuscript documents

`paper.md` is the single source of truth. To (re)generate the submission documents:

```bash
python3 qaip.py        # (re)generate all figures/tables/results
python3 svg2png.py     # rasterize figures SVG -> PNG
python3 build_docs.py  # writes paper.docx, paper.pdf, paper.html (figures embedded)
```

### Convert to PDF yourself (easiest, no tools needed)

Open **`paper.html`** in any web browser and choose **Print → Save as PDF**
(set margins to "Default" and enable "Background graphics"). The file is fully
self-contained — both figures are embedded as base64 images — so the PDF needs
no other files. This is the most reliable way to produce a clean, shareable PDF
on any machine.

All three scripts use only the Python standard library (the `.docx` is authored
directly as OOXML; the PDF and PNG are written by built-in minimal writers), so
they work fully offline with no pandoc/LibreOffice/LaTeX/matplotlib dependency.

The LaTeX version (`paper.tex`) targets the Elsevier `elsarticle` class and
compiles with `pdflatex paper.tex` in any standard TeX Live installation.

## Submission package

`submission/KBS_submission.zip` bundles every required item — manuscript
(`.docx`/`.pdf`/`.tex`), highlights, cover letter, both figures (PNG + SVG),
the graphical abstract, and the data tables — organized into labelled folders
with a `MANIFEST.txt`.

## Run it

No third-party packages are required — **Python 3.10+ standard library only**
(no numpy / pandas / scikit-learn / matplotlib). This maximises reproducibility
and auditability for reviewers.

```bash
python3 qaip.py
```

The run is fully deterministic (fixed seed `20260101`) and regenerates every
figure and table in `artifacts/` in ~20 seconds.

## Figures

The two figures are emitted as **SVG** (scalable vector graphics). Elsevier
accepts vector artwork (EPS/PDF) and SVG converts losslessly to PDF/EPS at
production; being vector, it satisfies any DPI requirement.

- **Figure 1** — convergence of the QIEA vs. classical GA and synergy-blind references.
- **Figure 2** — budget–risk-reduction efficiency frontier across budget levels.

## Tables

- **Table 1** — method comparison on the synergy-aware objective (20-seed means).
- **Table 2** — layer-wise ablation of the prediction, knowledge, and synergy layers.

## License

Released for academic use. See the manuscript for citation details.
