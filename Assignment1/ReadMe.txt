General description:
We take each movie title from the initial CSV file 'movievalue.csv' and for each of them we do
a REST call to the OMDB API, where we ask for more info about that movie. Because this is a very
time consuming operation, we only do this for about 3000 movies (the assignment said at leat 1000).

Namely, we are interested in info about Genre, IMDB Rating, IMDB Votes and Director. Also, because in
the initial file there is a lot of misssing info about release date, budget and popularity, we try
and replace this 'missingness' with the info provided by OMDB, when possible.

After getting this extra information, we 'clean' the data by removing all the movies we couldn't find
information about on OMDB (or about which we didn't try), so we are left with a data rich CSV.


How to run:
We used R (version 3.4.1) and RStudio (version 1.0.153).

Run 'package_install.R' to install the packages we use in the main script ('httr' for doing REST
calls and 'jsonlite' for working with JSONs, "ggplot2" for plots and "dplyr" and "reshape2" for
data manipulation).

In 'config.R' set the full paths of the 4 CSV files: the initial CSV, the enriched CSV, the 
cleaned CSV and the CSV that groups production houses by genres(the last one is for the
bonus part).

Run 'enrich_script.R' in order to get all the data from OMDB. This will take a few minutes.

Run 'clean_script.R' in order to clean the data.

Run 'figures.R' to get the plots.

Run 'imdb_vs_tomato.R' and 'genre_vs_production.R' for the bonus part.





The report is called 'main.pdf'.

