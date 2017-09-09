library(httr)
library(jsonlite)

#The file paths must be changed by the user
#initial_file_path <- "C:/MyData/School/Year1Sem1/IDS/Assignments/movievalue.csv"
#enriched_file_path <- "C:/MyData/School/Year1Sem1/IDS/Assignments/enrichedmovies.csv"


initial_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/movievalue.csv"
enriched_file_path <- "/home/xu/Documents/Intro to Data Science/Assignment1/team-07/Assignment1/enrichedmovies.csv"

#Reading the initial CSV in a data frame object and adding some blank fields
table <- read.csv(file=initial_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)
new_columns <- c("Genre", "IMDBRating", "IMDBVotes", "Director", "RottenTomatoesRating")
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
  #if (content(response)$Response){ BOGDAN IF
  if (is.null(content(response)$Response) == FALSE){
    found_movies <- found_movies + 1
    print(found_movies)
    #TengXu94: I need this if check movie 21 does not have the Genre field -> replacement has length zero error!
    if (length(content(response)$Genre) > 0){
      table[i,6] <- content(response)$Genre
    }
    if (length(content(response)$Ratings) > 0){
      table[i,7] <- content(response)$Ratings[[1]]$Value
      if (length(content(response)$Ratings)>1){
        table[i,10] <- content(response)$Ratings[[2]]$Value
      }
    }
    #TengXu94: I put if control also here because for some movie they do not have those data, just in case
    if (length(content(response)$imdbVotes) > 0){
      table[i,8] <- content(response)$imdbVotes
    }
    if (length(content(response)$Director) > 0){
      table[i,9] <- content(response)$Director
    }
    
    # If the revenue info is empty, then we replace it with the revenue info from
    # OMDB API
    if (table[i,5] == 0){
      if (identical(content(response)$BoxOffice,"N/A" == FALSE)){
        table[i,5] <- content(response)$BoxOffice
      }
    }
    
    #TengXu94: Same for release date!
    
    
  }
  if (found_movies > 1999){
    break;
  }
}

# Save the enriched data in a new list
write.csv(file=enriched_file_path, x=table)
