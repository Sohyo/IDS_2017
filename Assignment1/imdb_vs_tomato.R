require(ggplot2)
require(reshape2)

enriched_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/enrichedmovies.csv"

working_data <- read.csv(file=enriched_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)

for (i in 1:NROW(working_data)){
  
    working_data[i,8] <- gsub("/10", "", working_data[i,8] )
}

for (i in 1:NROW(working_data)){
  working_data[i,11] <- gsub("%", "", working_data[i,11] )
  working_data[i,11] <- gsub("/100", "", working_data[i,11] )
}

working_data2 <- working_data[!is.na(working_data$RottenTomatoesRating),]

plot(x = as.numeric(working_data2$IMDBRating),y = as.numeric(working_data2$RottenTomatoesRating),
     xlab = "IMDBRatings",
     ylab = "RTRatings",
     xlim = c(0,10),
     ylim = c(0,100),		 
     main = "IMDB ratings vs Rotten Tomatoes Ratings"
)