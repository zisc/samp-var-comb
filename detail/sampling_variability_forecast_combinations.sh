#!/bin/bash

# Abort if any commands fail.
set -e

# Set the environment variable OMP_NUM_THREADS to the number of physical CPU cores.
# In the docker container, this speads up pytorch computation by a factor of four.
# Presumably pytorch/openmp incorrectly set the number of threads by default on occasion.
export OMP_NUM_THREADS=$(grep '^core id' /proc/cpuinfo | sort -u | wc -l)

PAPER="sampling_variability_forecast_combinations"
PRESENTATION="presentation_sampling_variability_forecast_combinations"

Rscript --vanilla plot_simulation.R
Rscript --vanilla plot_SP500.R

# Compile visualisations produced by above R scripts.
(cd figure ; ./tex_to_pdf.sh)

# Compile paper to get intermediary output for bibtex to produce citations.
pdflatex -interaction=nonstopmode "${PAPER}.tex"

# Produce citations.
bibtex "${PAPER}"
bibtex supp

# Compile another three times for citations and references to labels to proliferate
# throughout document.
pdflatex -interaction=nonstopmode "${PAPER}.tex"
pdflatex -interaction=nonstopmode "${PAPER}.tex"
pdflatex -interaction=nonstopmode "${PAPER}.tex"

# Prepare submission for arXiv, see "https://arxiv.org/help/submit_tex".
tar --create --gzip --file=arxiv_submission.tar.gz "${PAPER}.tex" "${PAPER}.bbl" supp.bbl ECA_jasa.bst figure/FIG*.pdf figure/TBL*.tex

# Repeat above compilation process for ${PRESENTATION}.
pdflatex -interaction=nonstopmode "${PRESENTATION}.tex"
bibtex "${PRESENTATION}"
pdflatex -interaction=nonstopmode "${PRESENTATION}.tex"
pdflatex -interaction=nonstopmode "${PRESENTATION}.tex"
pdflatex -interaction=nonstopmode "${PRESENTATION}.tex"

