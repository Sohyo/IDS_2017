library(readr)
library(ggplot2)
library(dplyr)
source("config.R")

#Problem with read_tsv -> when the parser is expecting a character but the actual value is an integer
#it gives a "parsing failure" warning, use problems(data frame) in order to see which line cause it.
data <- read_tsv(file=dataset, col_names = c('userid', 'ts','artistid', 'artname', 'trid', 'trname'))
d_trimmed <- data[, c(grep("userid", colnames(data), value = TRUE),grep("artname", colnames(data), value = TRUE))]
sum(is.na(d_trimmed[2]))
# summary(d_trimmed)
