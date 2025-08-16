This project is a one-script pipeline for NYC traffic counts. It:

Downloads data from NYC Open Data’s API
Cleans it (clear column names, correct types, make a timestamp)
Saves both raw and clean CSVs and a SQLite database
Analyzes it to find the Top 10 busiest road segments 
Exports results as CSVs and charts in the output/ folder
