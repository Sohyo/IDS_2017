library(httr)
library(jsonlite)

#The file paths must be changed by the user
initial_file_path <- "C:/MyData/School/Year1Sem1/IDS/Assignments/movievalue.csv"
enriched_file_path <- "C:/MyData/School/Year1Sem1/IDS/Assignments/enrichedmovies.csv"

#Reading the initial CSV in a data frame object and adding some blank fields
table <- read.csv(file=initial_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)
new_columns <- c("Genre", "IMDBRating", "IMDBVotes", "Director")
for (i in new_columns){
  table[,i] <- NA
}

# This is a counter of the movies we find.
# We only want to get data on 2000 movies 
found_movies <- 0

# We iterate through the movies and get extra data from the OMDB API
for (i in 1:NROW(table)){
  title <- table[i, 1]
  print(title)
  response <- GET("http://www.omdbapi.com/?",
           query = list("apikey" = "863c5282", "t" = title)
  )
  if (content(response)$Response){
    found_movies <- found_movies + 1
    print(found_movies)
    table[i,6] <- content(response)$Genre
    if (length(content(response)$Ratings) > 0){
      table[i,7] <- content(response)$Ratings[[1]]$Value
    }
    table[i,8] <- content(response)$imdbVotes
    table[i,9] <- content(response)$Director
    
    # If the revenue info is empty, then we replace it with the revenue info from
    # OMDB API
    if (table[i,5] == 0){
      if (identical(content(response)$BoxOffice,"N/A" == FALSE)){
        table[i,5] <- content(response)$BoxOffice
      }
    }
  }
  if (found_movies > 1999){
    break;
  }
}

# Save the enriched data in a new list
write.csv(file=enriched_file_path, x=table)
