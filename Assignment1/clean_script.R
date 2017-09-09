source("config.R")

table <- read.csv(file=enriched_file_path, header=TRUE, sep=",", stringsAsFactors = FALSE)

#Delete rows with movies we couldn't find on OMDB
table<-table[!(is.na(table$Genre)),]

# Save the enriched data in a new list
write.csv(file=cleaned_file_path, x=table)
