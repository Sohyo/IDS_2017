library(readr)
library(ggplot2)
library(dplyr)
library(arules)
source("config.R")

#Problem with read_tsv -> when the parser is expecting a character but the actual value is an integer
#it gives a "parsing failure" warning, use problems(data frame) in order to see which line cause it.
data <- read_tsv(file=dataset, col_names = c('userid', 'ts','artistid', 'artname', 'trid', 'trname'))
#We'll live just the 2 columns that interest us
d_trimmed <- data[, c(grep("userid", colnames(data), value = TRUE),grep("artname", colnames(data), value = TRUE))]

# Note to the TAs/ Professor: You should have really put in the assignment the
# full name, not just "Beethoven" because it was pretty much a guessing game
# searching for the exact spelling.
d_beethoven <- subset(d_trimmed, artname == "Ludwig Van Beethoven") 
d_beethoven <- unique(d_beethoven) #all the users that listened to smth by Beethoven
d_interst <- subset(d_trimmed, userid %in% d_beethoven$userid)

# write.csv(d_trimmed, file = "MyData.csv")

sample_df <- d_interst[1:500000, ]

sample_df <- unique(sample_df)
trans <- as(split(sample_df$artname, sample_df[, "userid"] ), "transactions")
frequentItems <- eclat (trans, parameter = list(supp = 0.7, maxlen = 1))
inspect(frequentItems)

# rules <- apriori(trans, parameter = list(supp = 0.01, conf = 0.7, target = "rules"))
# as(rules, "data.frame");
# rules