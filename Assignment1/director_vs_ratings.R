require(ggplot2)
require(reshape2)

enriched_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/enrichedmovies.csv"

working_data <- read.csv(file=enriched_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)


# Obtain first genre in genres like in figures.R
vect = rep("", NROW(movies))
for (i in 1:NROW(movies)){
  genres = as.character(movies[i,'Genre'])
  first = strsplit(genres,",")[[1]][1]
  vect[i] <- first
}

working_data["Genre_first"] <- vect

#Cleaning Column Production House from N/A
working_data2 <- working_data[working_data$Production != "N/A",]



plot(x = factor(working_data3$Director),y = as.numeric(working_data2$IMDBRating),
     xlab = "IMDBRatings",
     ylab = "Directors",
     ylim = c(0,10),		 
     main = "Director vs Ratings"
)