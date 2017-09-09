require(ggplot2)
require(reshape2)

enriched_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/enrichedmovies.csv"

working_data <- read.csv(file=enriched_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)

for (i in 1:NROW(working_data)){
  
  working_data[i,8] <- gsub("/10", "", working_data[i,8] )
}

working_data2 <- working_data[!is.na(working_data$Director),]
working_data3 <- working_data2[working_data2$Director != "N/A",]

unique_directors <- list()
directors <- vector(mode="list", length=NROW(working_data3))
names(directors) <- working_data3$Director
for(i in 1:length(directors)){
  if(!( i %in% unique_directors)){
    unique_directors <- c(unique_directors, directors[i])
  }
}
print(unique_directors)

plot(x = factor(working_data3$Director),y = as.numeric(working_data2$IMDBRating),
     xlab = "IMDBRatings",
     ylab = "Directors",
     ylim = c(0,10),		 
     main = "Director vs Ratings"
)