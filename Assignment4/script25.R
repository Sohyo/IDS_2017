library(readr)
library(ggplot2)
library(dplyr)
library(arules)
library(arulesViz)
source("config.R")

#15121996 many rows!
data <- read_tsv(file=dataset,
                 col_names = c('userid', 'ts','artistid', 'artname', 'trid', 'trname'))

#13 452 816 na entries deleted
complete_case <- data[complete.cases(data), ]

#select just 10% of the whole dataset at random
sampled <- complete_case[sample(nrow(complete_case), 1000000), ]

#We'll live just the 2 columns that interest us

trimmed <- sampled[, c(grep("userid", colnames(sampled), value = TRUE),grep("artname", colnames(sampled), value = TRUE))]

#Atomic components Problem: We have duplicates, e.g for the same userid we have duplicate artname

aggrData <- split(trimmed$artname,trimmed$userid)

listData <- list()
for (i in 1:length(aggrData)) {
  listData[[i]] <- as.character(aggrData[[i]][!duplicated(aggrData[[i]])])
}

#Generate Transactions
trans <- as(listData,"transactions")

#Generate Rules
rules <- apriori(trans, parameter = list(minlen=2, maxlen=2, support=0.01, confidence=0.1, target = "rules"))

#Get all the rules where Iron Maiden is on the right
rules.sub <- subset(rules, subset = rhs %in% "Iron Maiden")
head(inspect(sort(rules.sub, decreasing = TRUE, na.last = NA, by = "lift", order = FALSE)))
