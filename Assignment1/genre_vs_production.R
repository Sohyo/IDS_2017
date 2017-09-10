require(ggplot2)
require(reshape2)
library(dplyr)
source("config.R")

# enriched_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/enrichedmovies.csv"
# grouped_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/groupedGenrePerProduction.csv"
working_data <- read.csv(file=enriched_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)


# Obtain first genre in genres like in figures.R
vect = rep("", NROW(working_data))
for (i in 1:NROW(working_data)){
  genres = as.character(working_data[i,'Genre'])
  first = strsplit(genres,",")[[1]][1]
  vect[i] <- first
}


working_data["Genre_first"] <- vect

#Cleaning Column Production House from N/A
working_data2 <- working_data[!is.na(working_data$Production),]
working_data3 <- working_data2[working_data2$Production != "N/A",]

#Cleaning Column Genre_First From N/A
working_data4 <- working_data3[working_data3$Genre_first != "N/A",]

# Create column with only 10 highest frequency Production Houses because they are too many to plot!
v = sort(table(working_data4$Production))
v1 = tail(as.data.frame(v),10)
vect1 = rep("", NROW(working_data4))
for (i in 1:NROW(working_data4)){
  production = as.character(working_data4[i,'Production'])
  if (is.element(production, v1[,1])){
    vect1[i] <- production
  } else {
    vect1[i] <- NA
  }
}

working_data4["Top_Productions"] <- vect1

#Cleaning Column Genre_First From N/A
working_data5 <- working_data4[!is.na(working_data4$Top_Productions),]

df <- working_data5[,c(12,13)]

#Grouping By Production Counting Genres
grouped <- group_by(df, df$Production, df$Genre_first) %>%
  summarise(count = n())

write.csv(file=grouped_file_path, x=grouped)
