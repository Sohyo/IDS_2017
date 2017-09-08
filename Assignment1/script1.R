library(httr)
library(jsonlite)

initial_file_path <- "C:/MyData/School/Year1Sem1/IDS/Assignments/movievalue.csv"
url <- "http://www.omdbapi.com/?apikey=863c5282&t="

#just this time
# title <- "titanic"
# r <- GET("http://www.omdbapi.com/", 
#          query = list("apikey" = "863c5282", "t" = "titanic")
# )
content(r)
table <- read.csv(file=initial_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)
new_columns <- c("Genre", "IMDBRating", "IMDBVotes")
for (i in new_columns){
  table[,i] <- NA
}

for (i in 1:NROW(table)){
  title <- table[i, 1]
  print(title)
  # title_url <- gsub(" ","%20",title)
  # print(title_url)
  response <- GET("http://www.omdbapi.com/?",
           query = list("apikey" = "863c5282", "t" = title)
  )
  print(content(response))
  if (content(response)$Response){
    table[i,6] <- content(response)$Genre
    if (length(content(response)$Ratings) > 0){
      table[i,7] <- content(response)$Ratings[[1]]$Value
    }
    table[i,8] <- content(response)$imdbVotes
  }
  print(i)

}
# path <- paste(url, title, sep='')
# response <- GET(path)
# content(response)$Genre
