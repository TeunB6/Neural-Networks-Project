require(tidyverse)

gs <- read.csv(file.choose())

analyse.parameter <- function(group, order) {
  group = enquo(group)
  order = enquo(order)
  param <- gs %>%
    group_by(!!group) %>%
    slice_min(order_by=!!order, with_ties=F, na_rm = T, n=3)
  return(param)
}

lr.durr <- analyse.parameter(learning_rate, testing_loss_durr)
lr.prob <- analyse.parameter(learning_rate, testing_loss_prob)
hidden_size.durr <- analyse.parameter(hidden_size, testing_loss_durr)
hidden_size.prob <- analyse.parameter(hidden_size, testing_loss_prob)


#  summarise(max_acc=max(testing_accuracy), mean_acc=mean(testing_accuracy), 
#            mean_prob_loss=mean(testing_loss_prob, na.rm=T), min_prob_loss=min(testing_loss_prob),
 #           mean_durr_loss=mean(testing_loss_durr, na.rm=T), min_durr_loss=min(testing_loss_durr))
