library(readr)
library(ggplot2)
library(dplyr)
library(arules)
source("config.R")

data <- read_tsv(file=dataset, col_names = c('userid', 'ts','artistid', 'artname', 'trid', 'trname'))
# We'll live just the 2 columns that interest us
d_trimmed <- data[, c(grep("userid", colnames(data), value = TRUE),grep("artname", colnames(data), value = TRUE))]

# Note: The assignment should have stated the
# full name, not just "Beethoven" because it was pretty much a guessing game
# searching for the exact spelling.
d_beethoven <- subset(d_trimmed, artname == "Ludwig Van Beethoven")
# All the users that listened to smth by Beethoven
d_beethoven <- unique(d_beethoven) 
d_interst <- subset(d_trimmed, userid %in% d_beethoven$userid)

# There are too many and we need to do a sampling
sample_df <- d_interst[1:500000, ]
sample_df <- unique(sample_df)

trans <- as(split(sample_df$artname, sample_df[, "userid"] ), "transactions")
frequentItems <- eclat (trans, parameter = list(supp = 0.7, maxlen = 1))
inspect(sort(frequentItems, decreasing = TRUE, na.last = NA, by = "support", order = FALSE))

# We will now find the most frequent bands in the original dataset (a sample)
sample_df_full <- unique(d_trimmed[1:500000, ])
trans_full <- as(split(sample_df_full$artname, sample_df_full[, "userid"] ), "transactions")
frequentItems2 <- eclat (trans_full, parameter = list(supp = 0.6, maxlen = 1))
inspect(sort(frequentItems2, decreasing = TRUE, na.last = NA, by = "support", order = FALSE))
