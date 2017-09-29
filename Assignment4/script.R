library(readr)
library(ggplot2)
library(dplyr)

#Problem with read_tsv -> when the parser is expecting a character but the actual value is an integer
#it gives a "parsing failure" warning, use problems(data frame) in order to see which line cause it.
data <- read_tsv(file="/home/xu/Documents/Intro to Data Science/Assignment4/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"
                 , col_names = c('user-id', 'artist-id','artist-name', 'plays'))

#head(data)
#problems(data)

columns = colnames(data)
vector_na = c()
for (i in columns){
  vector_na[i] = sum(is.na(data[i]))
}

#x = NROW(data) -> 6771128

data2 <- data.frame(vector_na)
data2[2] <- columns
colnames(data2)[1] <- '#missing values'
colnames(data2)[2] <- 'column'

# Plot missing values per column
barplot(data2$`#missing values`, names.arg=data2$column, main='Number of NA per column', xlab='Column name', ylab='Frequency', col='black')

#Generate a new table without missing values
final <- data[complete.cases(data), ]

#final

final2 <- subset(final, plays > 2000)
final2
#scatter plot: IMDB Rating vs. Budget
ggplot(final2,aes(x=final2$`artist-name`,y=as.numeric(final2$plays)))+geom_point(size=1,color="blue")+
  ggtitle("How budgets turn into IMDB rating")+
  xlab("Budget (in millions of US dollars)")+ylab("IMDB Rating")+geom_smooth(method=lm)