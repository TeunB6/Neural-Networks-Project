require(tidyverse)

gs <- read.csv(file.choose())

lr <- gs %>%
  group_by(learning_rate, optimizer) %>%
  summarise(max_acc=max(testing_accuracy), mean_acc=mean(testing_accuracy), 
            mean_prob_loss=mean(testing_loss_prob), min_prob_loss=min(testing_loss_prob),
            mean_durr_loss=mean(testing_loss_durr), min_durr_loss=min(testing_loss_durr))
init_range <- gs %>%
  group_by(init_range_start) %>%
  summarise(max_acc=max(testing_accuracy), mean_acc=mean(testing_accuracy), 
            mean_prob_loss=mean(testing_loss_prob), min_prob_loss=min(testing_loss_prob),
            mean_durr_loss=mean(testing_loss_durr), min_durr_loss=min(testing_loss_durr))
hidden_size <- gs %>%
  group_by(hidden_size) %>%
  summarise(max_acc=max(testing_accuracy), mean_acc=mean(testing_accuracy), 
            mean_prob_loss=mean(testing_loss_prob), min_prob_loss=min(testing_loss_prob),
            mean_durr_loss=mean(testing_loss_durr), min_durr_loss=min(testing_loss_durr))
