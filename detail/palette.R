library(colorspace)

score_lines <- c(
  "One-Stage" = "solid",
  "Two-Stage" = "dashed",
  "Two-Stage - Weights Fixed at Limit Optimiser" = "dotdash",
  "Two-Stage - Weights Fixed at Limit Optimizer" = "dotdash",
  "Two-Stage - Constituents Fixed at Point Estimates" = "dotted",
  "Log Score" = "longdash",
  "Optimise Log Score" = "longdash",
  "Optimize Log Score" = "longdash",
  "Measure Log Score" = "longdash",
  "Censored Log Score" = "22",
  "Optimise Censored Log Score" = "22",
  "Optimize Censored Log Score" = "22",
  "Measure Censored Log Score" = "22",
  "Full Sample" = "solid",
  "Drop Lowest Score" = "longdash",
  "95\\% Confidence Interval" = "14"
)

score_shapes <- c(
  "One-Stage" = 1,
  "Two-Stage" = 0,
  "Two-Stage - Weights Fixed at Limit Optimiser" = 5,
  "Two-Stage - Weights Fixed at Limit Optimizer" = 5,
  "Log Score" = 3,
  "LS" = 3,
  "Optimise Log Score" = 3,
  "Optimize Log Score" = 3,
  "Optimise LS" = 3,
  "Optimize LS" = 3,
  "Censored Log Score" = 4,
  "$\\mathrm{CS}_{<20\\%}$" = 4,
  "Optimise Censored Log Score" = 4,
  "Optimize Censored Log Score" = 4,
  "Optimise $\\mathrm{CS}_{<20\\%}$" = 4,
  "Optimize $\\mathrm{CS}_{<20\\%}$" = 4
)

score_colour <- qualitative_hcl(6, palette = "Dark 3")
score_colour <- score_colour[c(1,3,5,5,2,4,6)]
# Uncomment one of these lines to simulate the corresponding form of colour vision deficiency.
# score_colour <- deutan(score_colour)
# score_colour <- protan(score_colour)
# score_colour <- tritan(score_colour)
names(score_colour) <- c(
  "One-Stage",
  "Two-Stage",
  "Two-Stage - Weights Fixed at Limit Optimiser",
  "Two-Stage - Weights Fixed at Limit Optimizer",
  "LS",
  "$\\mathrm{CS}_{<10\\%}$",
  "$\\mathrm{CS}_{<20\\%}$"
)

score_fill <- c(
  "One-Stage" = NA,
  "Two-Stage" = NA,
  "Two-Stage - Weights Fixed at Limit Optimiser" = NA,
  "Two-Stage - Weights Fixed at Limit Optimizer" = NA,
  "Log Score" = NA,
  "LS" = NA,
  "Optimise Log Score" = NA,
  "Optimize Log Score" = NA,
  "Optimise LS" = NA,
  "Optimize LS" = NA,
  "Censored Log Score" = NA,
  "$\\mathrm{CS}_{<20\\%}$" = NA,
  "Optimise Censored Log Score" = NA,
  "Optimize Censored Log Score" = NA,
  "Optimise $\\mathrm{CS}_{<20\\%}$" = NA,
  "Optimize $\\mathrm{CS}_{<20\\%}$" = NA
)

score_breaks <- c(
  "One-Stage",
  "Two-Stage",
  "Two-Stage - Weights Fixed at Limit Optimiser",
  "Two-Stage - Weights Fixed at Limit Optimizer",
  "Two-Stage - Constituents Fixed at Point Estimates",
  "Log Score",
  "LS",
  "Optimise Log Score",
  "Optimize Log Score",
  "Optimise LS",
  "Optimize LS",
  "Measure Log Score",
  "Measure LS",
  "Censored Log Score",
  "$\\mathrm{CS}_{<10\\%}$",
  "$\\mathrm{CS}_{<20\\%}$",
  "Optimise Censored Log Score",
  "Optimize Censored Log Score",
  "Optimise $\\mathrm{CS}_{<20\\%}$",
  "Optimize $\\mathrm{CS}_{<20\\%}$",
  "Measure Censored Log Score",
  "Measure $\\mathrm{CS}_{<20\\%}$",
  "Full Sample",
  "Drop Lowest Score",
  "95\\% Confidence Interval"
)
