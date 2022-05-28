library(tibble)

momentify <- function(results_sans_moments) {
  arrange <- dplyr::arrange
  group_by <- dplyr::group_by
  summarise <- dplyr::summarise
  full_join <- dplyr::full_join
  mutate <- dplyr::mutate
  select <- dplyr::select
  distinct <- dplyr::distinct
  
  
  moments <- results_sans_moments$detail %>%
    as_tibble() %>%
    arrange(Estimator, Optimise, Measure, `Sample Size`)
  
  moments <- moments %>%
    group_by(Estimator, Optimise, Measure, `Sample Size`) %>%
    summarise(`Average Score` = mean(Score), .groups = "drop") %>%
    full_join(results_sans_moments$dgp_scores, by = "Measure") %>%
    mutate(`Average Divergence` = DGPScore - `Average Score`) %>%
    full_join(moments, by = c("Estimator", "Optimise", "Measure", "Sample Size")) %>%
    mutate(res2 = (Score - `Average Score`)^2)
  
  moments <- moments %>%
    group_by(Estimator, Optimise, Measure, `Sample Size`) %>%
    summarise(Variance = mean(res2), .groups = "drop") %>%
    full_join(moments, by = c("Estimator", "Optimise", "Measure", "Sample Size")) %>%
    mutate(resres22 = (res2 - Variance)^2)
  
  moments <- moments %>%
    group_by(Estimator, Optimise, Measure, `Sample Size`) %>%
    summarise(var_res2 = mean(resres22), .groups = "drop") %>%
    full_join(moments, by = c("Estimator", "Optimise", "Measure", "Sample Size"))
  
  za <- qnorm(0.5*(1+results_sans_moments$simulation_confidence_level)) 
  moments <- moments %>%
    group_by(Estimator, Optimise, Measure, `Sample Size`) %>%
    summarise(N = n(), .groups = "drop") %>%
    full_join(moments, by = c("Estimator", "Optimise", "Measure", "Sample Size")) %>%
    select(Estimator, Optimise, Measure, `Sample Size`, `Average Score`, `Average Divergence`, Variance, var_res2, N) %>%
    distinct() %>%
    mutate(se_avg = sqrt(Variance/N), se_var = sqrt(var_res2/N)) %>%
    mutate(
      `Expected Score CI Low` = `Average Score` - za*se_avg,
      `Expected Score CI High` = `Average Score` + za*se_avg,
      `Expected Divergence CI Low` = `Average Divergence` - za*se_avg,
      `Expected Divergence CI High` = `Average Divergence` + za*se_avg,
      `Variance CI Low` = Variance - za*se_var,
      `Variance CI High` = Variance + za*se_var
    ) %>%
    select(
      Estimator,
      Optimise,
      Measure,
      `Sample Size`,
      `Average Score`,
      `Expected Score CI Low`,
      `Expected Score CI High`,
      `Average Divergence`,
      `Expected Divergence CI Low`,
      `Expected Divergence CI High`,
      Variance,
      `Variance CI Low`,
      `Variance CI High`
    ) %>%
    mutate(
      `Optimise` = paste("Optimise", `Optimise`),
      `Measure` = paste("Measure", `Measure`)
    ) %>%
    mutate(
      `Optimise` = factor(
        `Optimise`,
        levels = sort(unique(`Optimise`), decreasing = TRUE)
      ),
      `Measure` = factor(
        `Measure`,
        levels = sort(unique(`Measure`), decreasing = TRUE)
      )
    ) %>%
    arrange(Estimator, Optimise, Measure, `Sample Size`)
  
  results_sans_moments$moments <- moments
  
  return(results_sans_moments)
}
