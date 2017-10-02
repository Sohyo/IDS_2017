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
sampled <- complete_case[sample(nrow(complete_case), 1800000), ]

#We'll live just the 2 columns that interest us

trimmed <- sampled[, c(grep("userid", colnames(sampled), value = TRUE),grep("artname", colnames(sampled), value = TRUE))]
#trimmed <- sampled[,grep("artname", colnames(sampled), value = TRUE)]


#Atomic components Problem: We have duplicates, e.g for the same userid we have duplicate artname

aggrData <- split(trimmed$artname,trimmed$userid)

listData <- list()
for (i in 1:length(aggrData)) {
  listData[[i]] <- as.character(aggrData[[i]][!duplicated(aggrData[[i]])])
}

#Generate Transactions
trans <- as(listData,"transactions")

#Generate Frequent Itemset

frequentItems <- apriori(trans, parameter = list(minlen=1, support=0.05, target = "frequent itemsets"))

#Filter transactions in order to Obtain only the interesting ones
listFrequentItems <- as(items(frequentItems), "list")
filtered <- listFrequentItems[sapply(listFrequentItems, function(x) (x == 'Eminem' || x == 'Ludwig Van Beethoven'))]
trans <- as(filtered,"transactions")


#Generate Rules
rules <- apriori(trans, parameter = list(minlen=1, maxlen=2, support=0.001, confidence=0.5))

#Plottin graph
plot(rules, method="graph", control=list("items"))
