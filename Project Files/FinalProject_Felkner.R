# I will first import my datasets from baseball reference
# I will be using the HOF data from 2015-2024
# After tuning my algorithm I will apply it to the 2025 likely candidate
# data for fun, but I won't know how accurate my predictions are until next year

# load the necessary packages
library(rvest)
library(tidyverse)
library(xtable)

url_2015 <- "https://www.baseball-reference.com/awards/hof_2015.shtml"

webpage_2015 <- read_html(url_2015)

data_2015 <- webpage_2015 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2015 <- data_2015[[1]]

# Assign the first row as column names
colnames(data_2015) <- data_2015[1, ]

# And then remove the first row since it is just the names
data_2015 <- data_2015[-1, ]

# and repeat this process for the subsequent years

# 2016
url_2016 <- "https://www.baseball-reference.com/awards/hof_2016.shtml"

webpage_2016 <- read_html(url_2016)

data_2016 <- webpage_2016 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2016 <- data_2016[[1]]

# Assign the first row as column names
colnames(data_2016) <- data_2016[1, ]

# And then remove the first row since it is just the names
data_2016 <- data_2016[-1, ]

# 2017
url_2017 <- "https://www.baseball-reference.com/awards/hof_2017.shtml"

webpage_2017 <- read_html(url_2017)

data_2017 <- webpage_2017 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2017 <- data_2017[[1]]

# Assign the first row as column names
colnames(data_2017) <- data_2017[1, ]

# And then remove the first row since it is just the names
data_2017 <- data_2017[-1, ]

# 2018
url_2018 <- "https://www.baseball-reference.com/awards/hof_2018.shtml"

webpage_2018 <- read_html(url_2018)

data_2018 <- webpage_2018 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2018 <- data_2018[[1]]

# Assign the first row as column names
colnames(data_2018) <- data_2018[1, ]

# And then remove the first row since it is just the names
data_2018 <- data_2018[-1, ]

# 2019
url_2019 <- "https://www.baseball-reference.com/awards/hof_2019.shtml"

webpage_2019 <- read_html(url_2019)

data_2019 <- webpage_2019 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2019 <- data_2019[[1]]

# Assign the first row as column names
colnames(data_2019) <- data_2019[1, ]

# And then remove the first row since it is just the names
data_2019 <- data_2019[-1, ]

# 2020
url_2020 <- "https://www.baseball-reference.com/awards/hof_2020.shtml"

webpage_2020 <- read_html(url_2020)

data_2020 <- webpage_2020 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2020 <- data_2020[[1]]

# Assign the first row as column names
colnames(data_2020) <- data_2020[1, ]

# And then remove the first row since it is just the names
data_2020 <- data_2020[-1, ]

# 2021
url_2021 <- "https://www.baseball-reference.com/awards/hof_2021.shtml"

webpage_2021 <- read_html(url_2021)

data_2021 <- webpage_2021 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2021 <- data_2021[[1]]

# Assign the first row as column names
colnames(data_2021) <- data_2021[1, ]

# And then remove the first row since it is just the names
data_2021 <- data_2021[-1, ]

# 2022
url_2022 <- "https://www.baseball-reference.com/awards/hof_2022.shtml"

webpage_2022 <- read_html(url_2022)

data_2022 <- webpage_2022 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2022 <- data_2022[[1]]

# Assign the first row as column names
colnames(data_2022) <- data_2022[1, ]

# And then remove the first row since it is just the names
data_2022 <- data_2022[-1, ]

# 2023
url_2023 <- "https://www.baseball-reference.com/awards/hof_2023.shtml"

webpage_2023 <- read_html(url_2023)

data_2023 <- webpage_2023 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2023 <- data_2023[[1]]

# Assign the first row as column names
colnames(data_2023) <- data_2023[1, ]

# And then remove the first row since it is just the names
data_2023 <- data_2023[-1, ]

# 2024
url_2024 <- "https://www.baseball-reference.com/awards/hof_2024.shtml"

webpage_2024 <- read_html(url_2024)

data_2024 <- webpage_2024 %>%
  html_nodes(css = "#hof_BBWAA") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2024 <- data_2024[[1]]

# Assign the first row as column names
colnames(data_2024) <- data_2024[1, ]

# And then remove the first row since it is just the names
data_2024 <- data_2024[-1, ]

# Now I need to manually add a Hall of Fame dummy variable because on
# baseball reference it's indicated by being highlighted which doesn't
# translate through the web scraping
data_2015$HoF <- ifelse(row.names(data_2015) %in% c(1:4), 1, 0)
data_2016$HoF <- ifelse(row.names(data_2016) %in% c(1:2), 1, 0)
data_2017$HoF <- ifelse(row.names(data_2017) %in% c(1:3), 1, 0)
data_2018$HoF <- ifelse(row.names(data_2018) %in% c(1:4), 1, 0)
data_2019$HoF <- ifelse(row.names(data_2019) %in% c(1:4), 1, 0)
data_2020$HoF <- ifelse(row.names(data_2020) %in% c(1:2), 1, 0)
data_2021$HoF <- 0
data_2022$HoF <- ifelse(row.names(data_2022) %in% c(1:1), 1, 0)
data_2023$HoF <- ifelse(row.names(data_2023) %in% c(1:1), 1, 0)
data_2024$HoF <- ifelse(row.names(data_2024) %in% c(1:3), 1, 0)


# now I will vertically bind each yearly dataset into one combined dataset
combined_data <- rbind(data_2015, data_2016, data_2017, data_2018, data_2019,
                       data_2020, data_2021, data_2022, data_2023, data_2024)

# export as a csv for my README
write.csv(combined_data, "combined_data.csv", row.names = TRUE)

# I am creating this raw data to include in my README in case someone else
# wanted to do different cleaning steps than I did
combined_raw_data <- rbind(data_2015, data_2016, data_2017, data_2018, data_2019,
                           data_2020, data_2021, data_2022, data_2023, data_2024)

# export as a csv for my README
write.csv(combined_raw_data, "combined_raw_data.csv", row.names = TRUE)

#########################################################################
# Data cleaning

# First I need to rename some variables that currently have the same name for
# both hitter and pitcher statistics (to allow for tidyverse transformations)
names(combined_data)[names(combined_data) == "G"] <- c("Batter_G", "Pitcher_G")
names(combined_data)[names(combined_data) == "H"] <- c("Batter_H", "Pitcher_H")
names(combined_data)[names(combined_data) == "HR"] <- c("Batter_HR", "Pitcher_HR")
names(combined_data)[names(combined_data) == "BB"] <- c("Batter_BB", "Pitcher_BB")


# then I will remove the "X-" prefix and "HOF" suffix from player names
# X- means they are leaving the ballot that year
# and HOF means they made the Hall in a subsequent year
# but I need to remove them so that I can remove duplicate names
combined_data$Name <- gsub("HOF$", "", combined_data$Name)
combined_data$Name <- gsub("^X-", "", combined_data$Name)

# now I will remove the duplicates of the players who appear more than once
# note that my code keeps each recurring player only in their final year of
# appearance, which would either be the year they were inducted into the HoF
# or the year they ran out of eligibility (after 10 years)
combined_data <- combined_data %>%
  group_by(Name) %>%
  slice(n()) %>%
  ungroup

# Now I need to convert all my statistics variable from character to numeric
combined_data <- combined_data %>%
  mutate_at(vars(6:38), as.numeric)

# Then I will exclude pitchers by excluding all observations with a nonzero
# number of wins (note that position players occasionally pitch in blowout
# games but they would not get a win for this)
combined_data <- combined_data %>%
  filter(is.na(W) | W == 0)

# Then I am going to remove all the pitching statistics (since I am only 
# focused on hitting here) and other unwanted variables such as position summary
combined_data <- combined_data %>%
  select(-c(26:39))

combined_data <- combined_data %>%
  select(-c(1:5))

# Finally I need to convert HoF to a factor variable since it's what I am 
# predicting in my classification model
combined_data$HoF <- factor(combined_data$HoF, levels = c(0, 1))

# Then I will create a table of summary statistics for my paper
summary_stats <- sapply(combined_data, function(x) 
  c(Min. = min(x, na.rm = TRUE),
    Q1 = quantile(x, 0.25, na.rm = TRUE),
    Median = median(x, na.rm = TRUE),
    Mean = mean(x, na.rm = TRUE),
    Q3 = quantile(x, 0.75, na.rm = TRUE),
    Std.Dev. = sd(x, na.rm = TRUE)))

summary_df <- as.data.frame(t(summary_stats))
latex_table <- xtable(summary_df, digits = 2)
print(latex_table, include.rownames = TRUE, sanitize.text.function = function(x) x)

# And I will also create a visualization for my paper
ba_plot <- ggplot(combined_data, aes(x = BA, fill = factor(HoF))) +
  geom_histogram(position = "identity", alpha = 0.5) +
  labs(x = "Batting Average", y = "Count", fill = "Hall of Fame?") +
  scale_fill_manual(values = c("grey", "red"), labels = c("Not in HoF", "In HoF")) +
  theme_minimal()

war_plot <- ggplot(combined_data, aes(x = WAR, fill = factor(HoF))) +
  geom_histogram(position = "identity", alpha = 0.5) +
  labs(x = "Wins Above Replacement", y = "Count", fill = "Hall of Fame?") +
  scale_fill_manual(values = c("grey", "red"), labels = c("Not in HoF", "In HoF")) +
  theme_minimal()

hr_plot <- ggplot(combined_data, aes(x = Batter_HR, fill = factor(HoF))) +
  geom_histogram(position = "identity", alpha = 0.5) +
  labs(x = "Home Runs", y = "Count", fill = "Hall of Fame?") +
  scale_fill_manual(values = c("grey", "red"), labels = c("Not in HoF", "In HoF")) +
  theme_minimal()

# save visualizations as png
ggsave("ba_plot.png", ba_plot, bg = "white")
ggsave("war_plot.png", war_plot, bg = "white")
ggsave("hr_plot.png", hr_plot, bg = "white")
##############################################################################

# Now that I have dataset ready, time to create my random forest model

# Load required libraries
library(tidymodels)
library(randomForest)

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
split <- initial_split(combined_data, prop = 0.7, strata = HoF)
train_data <- training(split)
test_data <- testing(split)

# Create a model specification
rf_spec <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

# Create a recipe
rf_recipe <- recipe(HoF ~ ., data = train_data)

# Create a workflow
rf_workflow <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(rf_recipe)

# Create a parameter grid for tuning
rf_grid <- grid_regular(
  mtry(range = c(1, ncol(train_data) - 1)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Define the cross-validation folds
cv_folds <- vfold_cv(train_data, v = 5)

# Tune the model with cross-validation
rf_tuned <- tune_grid(
  rf_workflow,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(accuracy)
)

# Select the best model
best_rf <- select_best(rf_tuned, metric = "accuracy")

# Finalize the workflow with the best model
final_rf <- finalize_workflow(
  rf_workflow,
  best_rf
)

# Fit the final model
final_rf_fit <- fit(final_rf, data = train_data)

# Make predictions on test data
test_predictions <- predict(final_rf_fit, new_data = test_data) %>%
  bind_cols(test_data)

# Calculate accuracy
accuracy <- test_predictions %>%
  metrics(truth = HoF, estimate = .pred_class) %>%
  filter(.metric == "accuracy") %>%
  pull(.estimate)

# Calculate confusion matrix
conf_matrix <- test_predictions %>%
  conf_mat(truth = HoF, estimate = .pred_class)

# print my performance metrics
print(accuracy)
print(conf_matrix)

# export confidence matrix to latex
conf_matrix_latex <- xtable(conf_matrix$table)
rownames(conf_matrix_latex) <- c("Prediction 0", "Prediction 1")
colnames(conf_matrix_latex) <- c("Truth 0", "Truth 1")
print(conf_matrix_latex, type = "latex")
###############################################################################

# now that I've trained and tested my model, I will apply it on the 2025 - 2028
# Hall of Fame ballot eligible players

# 2025
url_2025 <- "https://www.baseball-reference.com/awards/hof_2025.shtml"

webpage_2025 <- read_html(url_2025)

data_2025 <- webpage_2025 %>%
  html_nodes(css = "#hof_ballot") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2025 <- data_2025[[1]]

# Assign the first row as column names
colnames(data_2025) <- data_2025[1, ]

# And then remove the first row since it is just the names
data_2025 <- data_2025[-1, ]

# and I will perform some of the same cleaning steps I did before

# I need to rename some variables that currently have the same name for
# both hitter and pitcher statistics (to allow for tidyverse transformations)
names(data_2025)[names(data_2025) == "G"] <- c("Batter_G", "Pitcher_G")
names(data_2025)[names(data_2025) == "H"] <- c("Batter_H", "Pitcher_H")
names(data_2025)[names(data_2025) == "HR"] <- c("Batter_HR", "Pitcher_HR")
names(data_2025)[names(data_2025) == "BB"] <- c("Batter_BB", "Pitcher_BB")

# Now I need to convert all my statistics variable from character to numeric
data_2025 <- data_2025 %>%
  mutate_at(vars(5:37), as.numeric)

# Then I will exclude pitchers by excluding all observations with a nonzero
# number of wins (note that position players occasionally pitch in blowout
# games but they would not get a win for this)
data_2025 <- data_2025 %>%
  filter(is.na(W) | W == 0)

# and again remove the pitching and other unwanted variables
data_2025<- data_2025 %>%
  select(-c(25:38))

# and now that my data is prepared I can apply my model to predict on it
results_2025 <- predict(final_rf_fit, new_data = data_2025) 

# and then I will just combine the predictions with the Player Name for a 
# nice data frame
names_2025 <- data_2025$Name
predictions_2025 <- cbind(names_2025, results_2025)
colnames(predictions_2025) <- c("Name", "HoF Prediction")

# 2026
url_2026 <- "https://www.baseball-reference.com/awards/hof_2026.shtml"

webpage_2026 <- read_html(url_2026)

data_2026 <- webpage_2026 %>%
  html_nodes(css = "#hof_ballot") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2026 <- data_2026[[1]]

# Assign the first row as column names
colnames(data_2026) <- data_2026[1, ]

# And then remove the first row since it is just the names
data_2026 <- data_2026[-1, ]

# and I will perform some of the same cleaning steps I did before

# I need to rename some variables that currently have the same name for
# both hitter and pitcher statistics (to allow for tidyverse transformations)
names(data_2026)[names(data_2026) == "G"] <- c("Batter_G", "Pitcher_G")
names(data_2026)[names(data_2026) == "H"] <- c("Batter_H", "Pitcher_H")
names(data_2026)[names(data_2026) == "HR"] <- c("Batter_HR", "Pitcher_HR")
names(data_2026)[names(data_2026) == "BB"] <- c("Batter_BB", "Pitcher_BB")

# Now I need to convert all my statistics variable from character to numeric
data_2026 <- data_2026 %>%
  mutate_at(vars(5:37), as.numeric)

# Then I will exclude pitchers by excluding all observations with a nonzero
# number of wins (note that position players occasionally pitch in blowout
# games but they would not get a win for this)
data_2026 <- data_2026 %>%
  filter(is.na(W) | W == 0)

# and again remove the pitching and other unwanted variables
data_2026<- data_2026 %>%
  select(-c(25:38))

# and now that my data is prepared I can apply my model to predict on it
results_2026 <- predict(final_rf_fit, new_data = data_2026) 

# and then I will just combine the predictions with the Player Name for a 
# nice data frame
names_2026 <- data_2026$Name
predictions_2026 <- cbind(names_2026, results_2026)
colnames(predictions_2026) <- c("Name", "HoF Prediction")

# 2027
url_2027 <- "https://www.baseball-reference.com/awards/hof_2027.shtml"

webpage_2027 <- read_html(url_2027)

data_2027 <- webpage_2027 %>%
  html_nodes(css = "#hof_ballot") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2027 <- data_2027[[1]]

# Assign the first row as column names
colnames(data_2027) <- data_2027[1, ]

# And then remove the first row since it is just the names
data_2027 <- data_2027[-1, ]

# and I will perform some of the same cleaning steps I did before

# I need to rename some variables that currently have the same name for
# both hitter and pitcher statistics (to allow for tidyverse transformations)
names(data_2027)[names(data_2027) == "G"] <- c("Batter_G", "Pitcher_G")
names(data_2027)[names(data_2027) == "H"] <- c("Batter_H", "Pitcher_H")
names(data_2027)[names(data_2027) == "HR"] <- c("Batter_HR", "Pitcher_HR")
names(data_2027)[names(data_2027) == "BB"] <- c("Batter_BB", "Pitcher_BB")

# Now I need to convert all my statistics variable from character to numeric
data_2027 <- data_2027 %>%
  mutate_at(vars(5:37), as.numeric)

# Then I will exclude pitchers by excluding all observations with a nonzero
# number of wins (note that position players occasionally pitch in blowout
# games but they would not get a win for this)
data_2027 <- data_2027 %>%
  filter(is.na(W) | W == 0)

# and again remove the pitching and other unwanted variables
data_2027<- data_2027 %>%
  select(-c(25:38))

# and now that my data is prepared I can apply my model to predict on it
results_2027 <- predict(final_rf_fit, new_data = data_2027) 

# and then I will just combine the predictions with the Player Name for a 
# nice data frame
names_2027 <- data_2027$Name
predictions_2027 <- cbind(names_2027, results_2027)
colnames(predictions_2027) <- c("Name", "HoF Prediction")

# 2028
url_2028 <- "https://www.baseball-reference.com/awards/hof_2028.shtml"

webpage_2028 <- read_html(url_2028)

data_2028 <- webpage_2028 %>%
  html_nodes(css = "#hof_ballot") %>%
  html_table(fill = TRUE)

# This produces a list which we can turn into a table
data_2028 <- data_2028[[1]]

# Assign the first row as column names
colnames(data_2028) <- data_2028[1, ]

# And then remove the first row since it is just the names
data_2028 <- data_2028[-1, ]

# and I will perform some of the same cleaning steps I did before

# I need to rename some variables that currently have the same name for
# both hitter and pitcher statistics (to allow for tidyverse transformations)
names(data_2028)[names(data_2028) == "G"] <- c("Batter_G", "Pitcher_G")
names(data_2028)[names(data_2028) == "H"] <- c("Batter_H", "Pitcher_H")
names(data_2028)[names(data_2028) == "HR"] <- c("Batter_HR", "Pitcher_HR")
names(data_2028)[names(data_2028) == "BB"] <- c("Batter_BB", "Pitcher_BB")

# Now I need to convert all my statistics variable from character to numeric
data_2028 <- data_2028 %>%
  mutate_at(vars(5:37), as.numeric)

# Then I will exclude pitchers by excluding all observations with a nonzero
# number of wins (note that position players occasionally pitch in blowout
# games but they would not get a win for this)
data_2028 <- data_2028 %>%
  filter(is.na(W) | W == 0)

# and again remove the pitching and other unwanted variables
data_2028 <- data_2028 %>%
  select(-c(25:38))

# and now that my data is prepared I can apply my model to predict on it
results_2028 <- predict(final_rf_fit, new_data = data_2028) 

# and then I will just combine the predictions with the Player Name for a 
# nice data frame
names_2028 <- data_2028$Name
predictions_2028 <- cbind(names_2028, results_2028)
colnames(predictions_2028) <- c("Name", "HoF Prediction")

# Results are roughly what I would have expected, except that I am very 
# surprised to see Ichiro not predicted by my model

# Most baseball analysts think he is a lock for a near 100% vote so this 
# is very interesting

# Also not surprised that my model predicted A-Rod based off only his statistics,
# but he will most likely never be inducted due to his PED controversy

# I am not too surprised to see Pujols as the only other prediction in 
# these years, because most of the baseball analysts that I follow think 2025
# through 2027 are weak ballot years (with the exception of Ichiro which again
# I am surprised about)




