library(arules)
source("config.R")

df <- data.frame(
  TID = c(1,1,1,1,2,2,2,2,3,3,3,3,4,4,5,5,6,6), 
  item=c("1","2","3","4","2","3","4","5","3","4","5","6","2","7","3","5","5","6")
)
trans <- as(split(df[,"item"], df[,"TID"]), "transactions")
inspect(trans)

rules <- apriori(trans, parameter = list(supp = 0.3, conf = 0.7, target = "rules"))
as(rules, "data.frame");
rules
# summary(rules)