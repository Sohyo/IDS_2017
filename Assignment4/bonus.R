library(readr)
library(ggplot2)
library(dplyr)
library(arules)
library(arulesViz)
source("config.R")

#15121996 many rows!
data <- read_tsv(file="/home/xu/Documents/Intro to Data Science/Assignment4/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv",
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


plot(head(sort(rules, by = "lift"),20), method="graph", control=list("items"))


plot(tail(sort(rules, by = "confidence"),30), method="graph", control=list("items"))

plot(head(sort(rules, by = "confidence"),30), method="graph", control=list("items"))
