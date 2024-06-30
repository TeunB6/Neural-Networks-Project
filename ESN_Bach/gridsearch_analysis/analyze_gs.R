library(tidyverse)
library(ggplot2)
library(patchwork)

# Load the dataset (assuming you've already loaded it somehow)
gs <- read.csv(file.choose())
analyse_parameter <- function(group) {
  group <- enquo(group)
  param <- gs %>%
    mutate(across(where(is.numeric) & !matches("loss") & !matches("acc"), as.factor)) %>%
    group_by(!!group) %>%
    summarise(
      max_acc = max(testing_accuracy),
      mean_acc = mean(testing_accuracy),
      mean_prob_loss = mean(testing_loss_prob, na.rm = TRUE),
      min_prob_loss = min(testing_loss_prob, na.rm = TRUE),
      mean_durr_loss = mean(testing_loss_durr, na.rm = TRUE),
      min_durr_loss = min(testing_loss_durr, na.rm = TRUE)
    ) 
  return(param)
}

analyse_min <- function(group, order) {
  group <- enquo(group)
  order <- enquo(order)
  param <- gs %>%
    mutate(across(where(is.numeric) & !matches("loss") & !matches("acc"), as.factor)) %>%
    group_by(!!group) %>%
    slice_min(order_by = !!order, with_ties = F, na_rm = T, n = 3)
  
  return(param)
}

hidden.durr = analyse_min(hidden_size, testing_loss_durr)
hidden.prob = analyse_min(hidden_size, testing_loss_prob)
hidden.acc = analyse_min(hidden_size, -testing_accuracy)

# gs$init_range = as.factor(paste(as.character(gs$init_range_start), as.character(gs$init_range_end)))

# Analyze different parameters

# lr.data <- analyse_parameter(learning_rate)
# hidden.data <- analyse_parameter(hidden_size)
# weight_decay.data <- analyse_parameter(weight_decay)
# init_range.data <- analyse_parameter(init_range)

hidden.data <- analyse_parameter(hidden_size)
density.data <- analyse_parameter(density)
leakage_rate.data <- analyse_parameter(leakage_rate)
spectral_radius.data <- analyse_parameter(spectral_radius)
input_scaling.data <- analyse_parameter(input_scaling) 


# Function to generate plot
generate_plot <- function(data, x_var, y_var, title, x_lab, y_lab, show_legend = FALSE) {
  p <- ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point() +
    #geom_bar(stat = "identity", position = "dodge", show.legend = show_legend) +
    ggtitle(title) +
    labs(x=x_lab, y=y_lab) +
    theme_minimal()
  
  # Show legend on the bottom and centered for plot8 (weight_decay_prob)
  if (show_legend) {
    p <- p + theme(legend.position = "bottom", legend.justification = "center")
  } else {
    p <- p + theme(legend.position = "none")
  }
  
  return(p)
} 

# Generate plots with correct column names and manage legend display
plot1 <- generate_plot(density.data, "density", "min_durr_loss", "Density vs Duration Loss", "Density", "min MSE Loss", show_legend = F)
plot2 <- generate_plot(density.data, "density", "min_prob_loss", "Density vs Key Loss", "Density", "min Cross Entropy Loss", show_legend = F)
plot3 <- generate_plot(density.data, "density", "max_acc", "Density vs Accuracy", "Density", "min Accuracy", show_legend = F)

plot4 <- generate_plot(hidden.data, "hidden_size", "min_durr_loss", "Hidden Size vs Duration Loss", "Hidden Size", "min MSE Loss", show_legend = F)
plot5 <- generate_plot(hidden.data, "hidden_size", "min_prob_loss", "Hidden Size vs Key Loss", "Hidden Size", "min Cross Entropy Loss",  show_legend = F)
plot6 <- generate_plot(hidden.data, "hidden_size", "max_acc", "Hidden Size vs Accuracy", "Hidden Size", "min Accuracy", show_legend = F)

plot7 <- generate_plot(leakage_rate.data, "leakage_rate", "min_durr_loss", "Leakage Rate vs Duration Loss", "Leakage Rate", "min MSE Loss", show_legend = F)
plot8 <- generate_plot(leakage_rate.data, "leakage_rate", "min_prob_loss", "Leakage Rate vs Key Loss", "Leakage Rate", "min Cross Entropy Loss", show_legend = F)
plot9 <- generate_plot(leakage_rate.data, "leakage_rate", "max_acc", "Leakage Rate vs Key Accuracy", "Leakage Rate", "min Accuracy",  show_legend = F)

plot10 <- generate_plot(spectral_radius.data, "spectral_radius", "min_durr_loss", "Spectral Radius vs Duration Loss", "Spectral Radius", "min MSE Loss", show_legend = F)
plot11 <- generate_plot(spectral_radius.data, "spectral_radius", "min_prob_loss", "Spectral Radius vs Key Loss", "Spectral Radius", "min Cross Entropy Loss", show_legend = T)
plot12 <- generate_plot(spectral_radius.data, "spectral_radius", "max_acc", "Spectral Radius vs Accuracy", "Spectral Radius", "min Accuracy", show_legend = F)

plot13 <- generate_plot(input_scaling.data, "input_scaling", "min_durr_loss", "Input Scaling vs Duration Loss", "Input Scaling", "min MSE Loss", show_legend = F)
plot14 <- generate_plot(input_scaling.data, "input_scaling", "min_prob_loss", "Input Scaling vs Key Loss", "Input Scaling", "min Cross Entropy Loss", show_legend = T)
plot15 <- generate_plot(input_scaling.data, "input_scaling", "max_acc", "Input Scaling vs Accuracy", "Input Scaling", "min Accuracy", show_legend = F)

# Arrange plots in a grid using patchwork
((plot1 | plot2 | plot3) / (plot4 | plot5 | plot6) / (plot7 | plot8 | plot9) / (plot10 | plot11 | plot12) / (plot13 | plot14 | plot15))

