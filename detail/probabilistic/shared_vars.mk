export PROBABILISTIC_SILENCE_WARNS	:= "TRUE"
# Silences accepted warnings. This exists so that superfluous warnings may be
# turned on or of depending on whether we are testing (off) or whether the
# package is being installed (on). CRAN requires all warnings, even
# superflous ones, to be turned on and visible to their automated code auditor.

export R_HOME := $(shell Rscript --vanilla -e "cat(Sys.getenv('R_HOME'))")
export R_ARCH := $(shell Rscript --vanilla -e "cat(Sys.getenv('R_ARCH'))")
export R_INCLUDE_DIR := $(shell Rscript --vanilla -e "cat(R.home('include'))")
export R_BIN_DIR := $(shell Rscript --vanilla -e "cat(R.home('bin'))")
export R_SHARE_DIR := $(shell Rscript --vanilla -e "cat(R.home('share'))")

SRC=probabilistic/src

