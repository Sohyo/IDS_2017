library(httr)
library(jsonlite)
library(rvest)
library(xml2)

initial_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/movievalue.csv"
url <- "http://www.omdbapi.com/?apikey=863c5282&t="

content(r)
table <- read.csv(file=initial_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)
new_columns <- c("Genre", "IMDBRating", "IMDBVotes", "RatingTomatometer")
for (i in new_columns){
  table[,i] <- NA
}

#for (i in 1:NROW(table)){
for (i in 1:10){
  title <- table[i, 1]
  print(title)
  # title_url <- gsub(" ","%20",title)
  # print(title_url)
  response <- GET("http://www.omdbapi.com/",
           query = list("apikey" = "863c5282", "t" = title)
  )
  if (is.null(content(response)$Response) == FALSE){
    if (length(content(response)$Genre) > 0){
      table[i,6] <- content(response)$Genre
    }
    if (length(content(response)$Ratings) > 0){
      table[i,7] <- content(response)$Ratings[[1]]$Value
    }
    if (length(content(response)$imdbVotes) > 0){
      table[i,8] <- content(response)$imdbVotes
    }
  }
  View(table)
}
# path <- paste(url, title, sep='')
# response <- GET(path)
# content(response)$Genre
