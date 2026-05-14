# Practical: MSTAT-518-Data Science and Big Data Analytics
# Data Science using R
# Date: 14/05/2026
# ======================================
#
# Predicting House Sale Prices in Ames, Iowa
# ==========================================
#
# 1. Problem Understanding Phase

# No codes

# 2. Data Preparation Phase

# Load required libraries
library(tidyverse)
library(AmesHousing)   # Contains the Ames housing data

# Load data
ames <- make_ames()    # Preprocessed version of Ames housing data
# Alternative: raw data with more missing values available via ames_raw()

# Inspect structure
glimpse(ames)

# Check missing values (none in make_ames version, but for demonstration, we handle if any)
colSums(is.na(ames))

# Selecting variables without missing values
ames <- ames[, colSums(is.na(ames)) == 0]
dim(ames)

# Removing all missing values: case-wise
ames <- na.omit(ames)
dim(ames)

# Select relevant features (subset to avoid overplotting in EDA)
selected_features <- c("Sale_Price", "Gr_Liv_Area", "Year_Built", "Total_Bsmt_SF",
                       "Garage_Area", "Lot_Area", "Overall_Qual", "Overall_Cond",
                       "Central_Air", "Full_Bath", "Bedroom_AbvGr")

ames_subset <- ames %>% select(all_of(selected_features))

# Convert Central_Air to numeric 0/1
ames_subset <- ames_subset %>%
  mutate(Central_Air = ifelse(Central_Air == "Y", 1, 0))

# Check for any remaining missing values (should be none now)
sum(is.na(ames_subset))

hist(ames_subset$Sale_Price, main = "Histogram of Sale Price",
     xlab = "Sale Price")

hist(log(ames_subset$Sale_Price), main = "Histogram of Log of Sale Price",
     xlab = "Sale Price")

# Train-test split (80/20)
set.seed(123)
train_indices <- sample(1:nrow(ames_subset), size = 0.8 * nrow(ames_subset))
train_data <- ames_subset[train_indices, ]
test_data  <- ames_subset[-train_indices, ]

# 3. Exploratory Data Analysis Phase

# Summary statistics
summary(train_data)

# Distribution of sale price (target variable)
ggplot(train_data, aes(x = Sale_Price)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_log10(labels = scales::dollar) +
  labs(title = "Distribution of Sale Prices (log scale)", x = "Sale Price", y = "Count")

# Boxplot of Sale Price by Overall Quality (1 to 10 scale)
ggplot(train_data, aes(x = factor(Overall_Qual), y = Sale_Price)) +
  geom_boxplot(fill = "lightgreen") +
  scale_y_log10(labels = scales::dollar) +
  labs(title = "Sale Price by Overall Quality", x = "Overall Quality Rating", y = "Sale Price")

# Correlation matrix
cor_matrix <- train_data %>%
  select(-c(Sale_Price, Overall_Qual, Overall_Cond))%>%
  cor(use = "complete.obs")

# Load corrplot for visualization
library(corrplot)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Relationship between Gr_Liv_Area (above ground living area) and Sale_Price
ggplot(train_data, aes(x = Gr_Liv_Area, y = Sale_Price)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  scale_y_log10(labels = scales::dollar) +
  labs(title = "Sale Price vs. Living Area", x = "Living Area (sq ft)", y = "Sale Price")

# Key EDA Insights:
# •	Sale price is right-skewed → log transformation may help.
# •	Strong positive correlation: Gr_Liv_Area and Overall_Qual with Sale_Price.
# •	Weak correlation: Bedroom_AbvGr with price (price depends more on quality/area).

# Identify Outliers Using R
# Let the variable Total_Bsmt_SF: (Total Basement Square Feet)

# We obtain the standardized variable as follows:
z_Bsmt <- (ames_subset$Total_Bsmt_SF - mean(ames_subset$Total_Bsmt_SF))/
  sd(ames_subset$Total_Bsmt_SF)

(2300 - mean(ames_subset$Total_Bsmt_SF))/sd(ames_subset$Total_Bsmt_SF)

# 2300 sqft basement is not identified as an outlier using this method, since 2.83 < 3.

# The which() command identifies records that meet specified conditions.

Bsmt_outliers <- which(z_Bsmt < - 3 | z_Bsmt > 3)
Bsmt_outliers
length(Bsmt_outliers)

# Construct Contingency Tables Using R
# The command to create a table is table()

t.v1 <- table(ames_subset$Overall_Qual, 
              ames_subset$Bedroom_AbvGr)
t.v1
names(dimnames(t.v1)) <- c("Overall_Qual", "Bedroom_AbvGr")
t.v1

# To add row and column totals to the table, use the addmargins() command.
t.v2 <- addmargins(A = t.v1, FUN =list(total= sum), quiet=T)
t.v2

# Now we want to edit table t.v1 so it gives us the row percentages.
round(prop.table(t.v1, margin = 1)*100, 2)


Sale_Price_Cat <- ifelse(train_data$Sale_Price <= 
                           quantile(train_data$Sale_Price, 0.75), 
                         "Low","High")
freq.tab1 <- table(Sale_Price_Cat)
freq.tab1
barplot(freq.tab1, main="Sale Price Category", xlab="Category", ylab="Frequency")

train_data_cat <- cbind(Sale_Price_Cat, train_data)

to.resample <- which(Sale_Price_Cat == "High")
our.resample <- sample(x = to.resample, 
                       size = 590, replace = TRUE)

our.resample <- train_data_cat[our.resample, ]

train_data_rebal <- rbind(train_data_cat, our.resample)

freq.tab2 <- table(train_data_rebal$Sale_Price_Cat)
freq.tab2
barplot(freq.tab2, main="Sale Price Category Balanced", xlab="Category", ylab="Frequency")
c(freq.tab2[1]/sum(freq.tab2), freq.tab2[2]/sum(freq.tab2))


# 4. Setup Phase

# Log transform Sale_Price on both train and test
train_data <- train_data %>%
  mutate(log_Sale_Price = log(Sale_Price))

test_data <- test_data %>%
  mutate(log_Sale_Price = log(Sale_Price))

# Define model formula (using log price as target)
formula <- log_Sale_Price ~ Gr_Liv_Area + Year_Built + 
  Total_Bsmt_SF + Garage_Area + Lot_Area + 
  Overall_Qual + Overall_Cond + Central_Air + 
  Full_Bath + Bedroom_AbvGr

# Setup cross-validation (5-fold)
library(caret)
set.seed(456)
train_control <- trainControl(method = "cv", number = 5, 
                              verboseIter = FALSE)

# 5. Modeling Phase

# Model 1: Multiple Linear Regression
lm_model <- train(formula, data = train_data, method = "lm",
                  trControl = train_control)

# Model 2: Random Forest
library(randomForest)
rf_model <- train(formula, data = train_data, method = "rf",
                  trControl = train_control, 
                  ntree = 100, tuneLength = 3)

# Print results
lm_model
summary(lm_model)
rf_model
summary(rf_model)

# 6. Evaluation Phase

# Function to compute RMSE on original price scale
rmse_orig <- function(model, newdata, actual_price_col = "Sale_Price") {
  pred_log <- predict(model, newdata = newdata)
  pred_price <- exp(pred_log)   # back-transform from log
  actual_price <- newdata[[actual_price_col]]
  sqrt(mean((pred_price - actual_price)^2))
}

# Evaluate both models
rmse_lm <- rmse_orig(lm_model, test_data)
rmse_rf <- rmse_orig(rf_model, test_data)

cat("Linear Regression RMSE: $", round(rmse_lm, 0), "\n")
cat("Random Forest RMSE: $", round(rmse_rf, 0), "\n")

# Visualize predictions vs actuals (on test set)
pred_lm <- exp(predict(lm_model, newdata = test_data))
pred_rf <- exp(predict(rf_model, newdata = test_data))

comparison <- test_data %>%
  select(Sale_Price) %>%
  mutate(LM_Pred = pred_lm, RF_Pred = pred_rf) %>%
  pivot_longer(cols = c(LM_Pred, RF_Pred), names_to = "Model", values_to = "Pred_Price")

ggplot(comparison, aes(x = Sale_Price, y = Pred_Price, color = Model)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_x_log10(labels = scales::dollar) +
  scale_y_log10(labels = scales::dollar) +
  labs(title = "Predicted vs Actual Sale Prices", x = "Actual Price", y = "Predicted Price")

# Feature importance (Random Forest)
importance_df <- varImp(rf_model)$importance
importance_df$Feature <- rownames(importance_df)
ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "", y = "Importance")

# 7. Deployment Phase

# We will save the best model and build a simple prediction function for deployment.
# Choose the best model (lower RMSE)
if(rmse_rf < rmse_lm) {
  final_model <- rf_model
  cat("Deploying Random Forest model.\n")
} else {
  final_model <- lm_model
  cat("Deploying Linear Regression model.\n")
}

# Save model to disk
saveRDS(final_model, file = "house_price_model.rds")

# Create a prediction function for new data
predict_price <- function(new_house_df) {
  # new_house_df must have the same column names as train_data (except Sale_Price)
  # Ensure Central_Air is numeric 0/1 if character
  if (is.character(new_house_df$Central_Air)) {
    new_house_df <- new_house_df %>%
      mutate(Central_Air = ifelse(Central_Air == "Y", 1, 0))
  }
  log_pred <- predict(final_model, newdata = new_house_df)
  return(exp(log_pred))
}

# Example usage with a single new house (values typical for Ames)
new_house <- tibble(
  Gr_Liv_Area = 1500,
  Year_Built = 2005,
  Total_Bsmt_SF = 750,
  Garage_Area = 400,
  Lot_Area = 8000,
  Overall_Qual = "Very_Good",
  Overall_Cond = "Good",
  Central_Air = "Y",
  Full_Bath = 2,
  Bedroom_AbvGr = 3
)

predicted_price <- predict_price(new_house)
cat("Predicted sale price for the new house: $", 
    round(predicted_price, 0), "\n")

# ===== END =======================
