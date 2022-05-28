library(ggplot2)

# We define an elsivier ggplot2 theme to format our plots to the publisher's requirements.

theme_elsivier_axis_text_size <- 6
theme_elsivier_axis_title_size <- 8
theme_elsivier_strip_text_size <- 7
theme_elsivier_legend_title_size <- theme_elsivier_axis_title_size
theme_elsivier_legend_text_size <- theme_elsivier_strip_text_size

theme_elsivier <- function() {
  theme_minimal() %+replace%
  theme(
    legend.position = "bottom",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA),
    axis.ticks = element_line(),
    axis.text = element_text(size = theme_elsivier_axis_text_size),
    axis.title = element_text(size = theme_elsivier_axis_title_size),
    strip.text = element_text(size = theme_elsivier_strip_text_size),
    strip.text.x = element_text(angle = 0, margin = margin(b = 0.75, unit = "mm")),
    strip.text.y = element_text(angle = 270, margin = margin(l = 0.75, unit = "mm")),
    legend.title = element_text(size = theme_elsivier_legend_title_size),
    legend.text = element_text(size = theme_elsivier_legend_text_size)
      # See
      # https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions/artwork-sizing
      # for font size instructions.
  )
}
  