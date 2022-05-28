#!/bin/bash
set -e
for n in {1..8}
do
    FIGURE="FIG${n}"
    if [[ ! -f "${FIGURE}.pdf" ]]; then
        lualatex --interaction=nonstopmode --jobname="FIG${n}" "FIG${n}.tex"
    fi
done

exit 0

