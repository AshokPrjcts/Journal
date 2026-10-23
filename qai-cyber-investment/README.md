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
├── build_docs.py     # converts paper.md -> paper.docx and paper.pdf (stdlib only)
├── paper.md          # the short-communication manuscript (source of truth)
├── paper.docx        # Word version (generated)
├── paper.pdf         # PDF version (generated)
├── highlights.md     # KBS highlights (separate submission file)
├── cover_letter.md   # editor cover letter
├── README.md         # this file
└── artifacts/        # generated on each run (figures, tables, results)
    ├── figure1_convergence.svg
    ├── figure2_frontier.svg
    ├── graphical_abstract.svg
    ├── table1_performance.csv / .md
    ├── table2_ablation.csv / .md
    └── results.json
```

## Generate the Word and PDF manuscript

`paper.md` is the single source of truth. To (re)generate the submission documents:

```bash
python3 build_docs.py      # writes paper.docx and paper.pdf
```

This uses only the Python standard library (a `.docx` is authored directly as
OOXML; the PDF is written by a built-in minimal writer), so it works offline
with no pandoc/LibreOffice/LaTeX dependency.

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
