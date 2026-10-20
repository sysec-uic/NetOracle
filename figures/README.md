# figures — paper-figure reproduction kit

Regenerates every plot in the paper from the JSON files in `data/`.
Only matplotlib is required.

    python3 plot_figures.py        # -> fig_phase.pdf, fig_analysis.pdf, fig_grounding.pdf
    pdflatex fig_listing.tex       # -> fig_listing.pdf (the query/gate listing figure)

- Style knobs live at the top of `plot_figures.py`: `PAL` is the paper's
  8-color palette, `GROUNDING_STYLE` maps each model to a color+marker,
  `PHASE_LABELS` names the phases in the daemon trace.
- `data/` holds plot-ready JSON. To refresh after new experiments:
  - phase / analysis: run `python3 scripts/make_figures_hotnets.py` at the
    repo root, then copy `results/figdata/*.json` here;
  - grounding: run `python3 export_grounding.py` here (reads
    `results/eval/` directly; keeps only the five in-context models plus
    the kNN reference).
