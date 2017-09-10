library(ggplot2)
source("config.R")


movies <- read.csv(cleaned_file_path, header=TRUE, sep=",")

# Histogram of Popularity
ggplot(movies,aes(Popularity))+geom_histogram(binwidth = 0.05, color="red", fill="blue")+
  xlim(0.05,5)+ggtitle("Histogram of movie popularity")

# Density plot of Popularity
ggplot(movies,aes(Popularity))+geom_density(color="red", fill="blue")+
  xlim(0.05,5)+ggtitle("Density plot of movie popularity")

# Density plot of Budget
ggplot(movies,aes(Budget))+geom_density(color="red", fill="blue")+
  ggtitle("Density plot of movie budgets")+xlim(100000,100000000)



# Obtain first genre in genres (usually has 3)
vect = rep("", NROW(movies))
for (i in 1:NROW(movies)){
  genres = as.character(movies[i,'Genre'])
  first = strsplit(genres,",")[[1]][1]
  vect[i] <- first
}

movies["Genre_first"] <- vect # Create extra column with first genre

# Create column with only 6 highest frequency genres
v = sort(table(movies$Genre_first))
v1 = tail(as.data.frame(v),6)
vect1 = rep("", NROW(movies))
for (i in 1:NROW(movies)){
  genre = as.character(movies[i,'Genre_first'])
  if (is.element(genre, v1[,1])){
    vect1[i] <- genre
  } else {
    vect1[i] <- NA
  }
}
movies["Top_genres"] <- vect1

# Bar plot of Most frequent genres
ggplot(subset(movies, !is.na(movies$Top_genres)), aes(Top_genres))+ geom_bar()+xlab("Most frequent genres")+
  ggtitle("Six most frequent genres")

# Popularity of most frequent genres
ggplot(subset(movies, !is.na(movies$Top_genres)),aes(Popularity, color=Top_genres))+geom_density()+
  xlim(0.5,5.7)+ggtitle("Popularity of most frequent genres.")


# Erase /10 from IMDB Ratings
vect = rep("", NROW(movies))
for (i in 1:NROW(movies)){
  rating = as.character(movies[i,'IMDBRating'])
  vect[i] <- gsub('/10','',rating)
}
movies["IMDBRating"] <- vect # Create extra column with first genre

# Budget vs. IMDB Rating
movies2 <- subset(movies, Budget > 100000)
movies2 <- subset(movies, !is.na(IMDBRating))


#scatter plot: IMDB Rating vs. Budget
ggplot(movies2,aes(x=Budget,y=as.numeric(IMDBRating)))+geom_point(size=1,color="blue")+
  ggtitle("How budgets turn into IMDB rating")+xlim(1000000,50000000)+ylim(3.5,9)+
  xlab("Budget (in millions of US dollars)")+ylab("IMDB Rating")+geom_smooth(method=lm)  


# Scatter plot of Budget and popularity
ggplot(movies2,aes(x=Budget,y=as.numeric(Popularity)))+geom_point(size=1,color="blue")+
  ggtitle("How budgets turn into Popularity.")+xlim(1000000,50000000)+
  xlab("Budget (in millions of US dollars)")+ylab("Popularity")+geom_smooth(method=lm)+ylim(0.1,6)


