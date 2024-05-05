# ECON 5253 Final Project
This project uses random forest machine learning models to prediction Baseball Hall of Fame induction. Directions for how to replicate the results are as follows

# Obtaining the data
My data can be obtained through web-scraping the baseball-reference website. Each annual dataset can be accessed by substituting the appropriate year into the following URL: https://www.baseball-reference.com/awards/hof_2015.shtml#all_hof_BBWAA. For 2024 and earlier years, the desired selector path is #hof_BBWAA. For 2025 and later years, the desired selector path is #hof_ballot. Running my R script in the Scripts folder will also do all of this for you, if you prefer.

Alternatively, a csv file of my combined and cleaned is available in the Data folder as combined_data.

Alternatively, if you wish to do your own cleaning of the combined data, a csv file of combined but uncleaned data is available in the Data folder as combined_raw_data.

# Cleaning the data
Lines 217-263 of my R script replicate the cleaning that I did on my test/train data. These cleaning procedures are repeated for prediction on future data (2025-2028) in lines 407-574 of my R script (note that this section of the script also includes importing and running predictions on each future dataset.)

# Running models
Lines 300-379 of my R script replicate the random forest model that I ran with my test/train data. 

# Creating tables
My summary statistics table can be replicated and exported to Latex code by running lines 266-276 of my R script. My confusion matrix can be replicated and exported to Latex code by running lines 368-379 of my R script. 

# Creating images



