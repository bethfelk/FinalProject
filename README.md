# ECON 5253 Final Project
This project uses random forest machine learning models to prediction Baseball Hall of Fame induction. Directions for how to replicate the results are as follows

# Obtaining the data
My data can be obtained through web-scraping the baseball-reference website. Each annual dataset can be accessed by substituting the appropriate year into the following URL: https://www.baseball-reference.com/awards/hof_2015.shtml#all_hof_BBWAA. For 2024 and earlier years, the desired selector path is #hof_BBWAA. For 2025 and later years, the desired selector path is #hof_ballot. Running my R script in the Scripts folder will also do all of this for you, if you prefer.

Alternatively, a csv file of my combined and cleaned is available in the Data folder as `combined_data`.

Alternatively, if you wish to do your own cleaning of the combined data, a csv file of combined but uncleaned data is available in the Data folder as `combined_raw_data`.

# Cleaning the data, running models, and creating tables/images
My data cleaning, running of models, and creations of tables and images can be replicated by running the `FinalProject_Felkner.R` script in the Project Files folder.



