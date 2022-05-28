latex_packages = c(
  "\\usepackage{amssymb}",
  "\\usepackage{mathtools}",
  "\\usepackage{mathrsfs}",
  "\\usepackage{tikz}"
)

tikz_default_engine = "pdftex"

options(
  tikzDefaultEngine = tikz_default_engine,
  tikzDocumentDeclaration = "\\documentclass[tikz]{standalone}",
  tikzLatexPackages = latex_packages,
  tikzLualatexPackages = latex_packages,
  tikzLwdUnit = 72.27/96
)

page_width_inches = 6.47572
page_height_inches = 8.60661
full_width_height_inches = 4.0
full_width_aspect_ratio = page_width_inches/full_width_height_inches
full_height_width_inches_landscape = page_height_inches/full_width_aspect_ratio

col_width_inches = 3.07181

letter_label_x_offset <- 0.07
letter_label_y_offset <- letter_label_x_offset * full_width_aspect_ratio
