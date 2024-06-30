library(tidyverse)
library(ggplot2)
library(patchwork)

# Load the dataset (assuming you've already loaded it somehow)
gs <- read.csv(file.choose())
analyse_parameter <- function(group, order) {
  group <- enquo(group)
  order <- enquo(order)
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
lr_durr <- analyse_parameter(learning_rate, testing_loss_durr)
lr_prob <- analyse_parameter(learning_rate, testing_loss_prob)
hidden_size_durr <- analyse_parameter(hidden_size, testing_loss_durr)
hidden_size_prob <- analyse_parameter(hidden_size, testing_loss_prob)
weight_decay_durr <- analyse_parameter(weight_decay, testing_loss_durr)
weight_decay_prob <- analyse_parameter(weight_decay, testing_loss_prob)
init_range_durr <- analyse_parameter(init_range, testing_loss_durr)
init_range_prob <- analyse_parameter(init_range, testing_loss_prob)

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
plot1 <- generate_plot(lr_durr, "learning_rate", "mean_durr_loss", "Learning Rate vs Duration Loss", "Learning Rate", "Mean MSE Loss", show_legend = F)
plot2 <- generate_plot(lr_prob, "learning_rate", "mean_prob_loss", "Learning Rate vs Key Loss", "Learning Rate", "Mean Cross Entropy Loss", show_legend = F)
plot3 <- generate_plot(lr_prob, "learning_rate", "mean_acc", "Learning Rate vs Accuracy", "Learning Rate", "Mean Accuracy", show_legend = F)

plot4 <- generate_plot(hidden_size_durr, "hidden_size", "mean_durr_loss", "Hidden Size vs Duration Loss", "Hidden Size", "Mean MSE Loss", show_legend = F)
plot5 <- generate_plot(hidden_size_prob, "hidden_size", "mean_prob_loss", "Hidden Size vs Key Loss", "Hidden Size", "Mean Cross Entropy Loss",  show_legend = F)
plot6 <- generate_plot(hidden_size_prob, "hidden_size", "mean_acc", "Hidden Size vs Accuracy", "Hidden Size", "Mean Accuracy", show_legend = F)

plot7 <- generate_plot(weight_decay_durr, "weight_decay", "mean_durr_loss", "Weight Decay vs Duration Loss", "Weight Decay", "Mean MSE Loss", show_legend = F)
plot8 <- generate_plot(weight_decay_prob, "weight_decay", "mean_prob_loss", "Weight Decay vs Key Loss", "Weight Decay", "Mean Cross Entropy Loss", show_legend = F)
plot9 <- generate_plot(weight_decay_prob, "weight_decay", "mean_acc", "Weight Decay vs Key Accuracy", "Weight Decay", "Mean Accuracy",  show_legend = F)

plot10 <- generate_plot(init_range_durr, "init_range", "mean_durr_loss", "Initialization Range vs Duration Loss", "Initialization Range", "Mean MSE Loss", show_legend = F)
plot11 <- generate_plot(init_range_prob, "init_range", "mean_prob_loss", "Initialization Range vs Key Loss", "Initialization Range", "Mean Cross Entropy Loss", show_legend = T)
plot12 <- generate_plot(init_range_prob, "init_range", "mean_acc", "Initialization Range vs Accuracy", "Initialization Range", "Mean Accuracy", show_legend = F)

# Arrange plots in a grid using patchwork
((plot1 | plot2 | plot3) / (plot4 | plot5 | plot6) / (plot7 | plot8 | plot9) / (plot10 | plot11 | plot12))

