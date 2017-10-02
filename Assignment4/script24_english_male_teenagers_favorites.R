library(readr)
library(ggplot2)
library(dplyr)
library(arules)
library(arulesViz)
source("config.R")

#15121996 many rows!
data <- read_tsv(file=dataset,
                 col_names = c('userid', 'ts','artistid', 'artname', 'trid', 'trname'))

data2 <- read_tsv(file=dataset_user, col_names = c('userid', 'gender', 'age', 'country', 'signup'))

#13 452 816 na entries deleted
complete_case <- data[complete.cases(data), ]

#select just 10% of the whole dataset at random
sampled <- complete_case[sample(nrow(complete_case), 3000000), ]
#We'll live just the 2 columns that interest us

trimmed <- sampled[, c(grep("userid", colnames(sampled), value = TRUE),grep("artname", colnames(sampled), value = TRUE))]
head(data2)
working_table <- merge(trimmed, data2, by = "userid")

#Some user do not have age
final_table <- working_table[complete.cases(working_table), ]
final_table$age <- as.numeric(final_table$age)
head(final_table)

#Ex3 english teenager + male -> band

#Pick only English Teenagers
newdata <- final_table[ which(final_table$gender=='m'
                              & final_table$age > 14 & final_table$age <20 & final_table$country=='United Kingdom'), ]


#Usual Algorithm
trimmed2 <- newdata[, c(grep("userid", colnames(newdata), value = TRUE),grep("artname", colnames(newdata), value = TRUE) )]

aggrData <- split(trimmed2$artname,trimmed2$userid)

listData <- list()
for (i in 1:length(aggrData)) {
  listData[[i]] <- as.character(aggrData[[i]][!duplicated(aggrData[[i]])])
}

#Generate Transactions
trans <- as(aggrData,"transactions")


frequentItems <- apriori(trans, parameter = list(minlen=1,maxlen=1, support=0.2, target = "frequent itemsets"))

inspect(head(sort(frequentItems,by='count'),10))