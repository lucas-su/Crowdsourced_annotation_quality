library(readr)
library(tidyverse)
library(brms)
library(stringr)

library(dplyr)
library(rstanarm)
library(mascutils)
library(gridExtra)
library(devtools)
library(knitr)
library(mascutils)
library(bayr)
library(naniar)
library(ggplot2)
library(tidybayes)

data <- read_csv("C:\\Users\\admin\\PycharmProjects\\Crowdsourced_annotation_quality\\exports\\mcmc_uniform.csv", 
                           col_names = c('car','dup','p_fo', 'p_kg', 'p_kg_u', 'u_error', 'pc_m', 'pc_n'))

lm <-  brm(u_error ~ car + dup + p_fo + p_kg + p_kg_u, 
                                       data = data,
                                       family = gaussian(),
                                       control = list(adapt_delta = 0.99),
                                       cores = 4, backend = "cmdstanr", threads = threading(4))

## coefs
coefs <- coef(posterior(lm), interval=0.9)
coefs
plot(lm)

predicted_draws(lm,data)


post = posterior_predict(lm)

post <- data.frame(post)

ggplot(aes(x = post)) +
  stat_slab()
