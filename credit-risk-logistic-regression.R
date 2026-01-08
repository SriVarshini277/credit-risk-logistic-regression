#load data
credit_data <- read.table("germancredit.txt", header = FALSE)
#check data
head(credit_data)

#The last column (V21) is our answer: 1 = good credit, 2 = bad credit
#Let's make it 1 = good credit risk, 0 = bad credit risk
credit_data$good_credit <- ifelse(credit_data$V21 == 1, 1, 0)

#Split into training and testing data
#We'll use 70% to train, 30% to test
set.seed(123)  
n <- nrow(credit_data)
train_indices <- sample(1:n, size = 0.7 * n)

train_data <- credit_data[train_indices, ]
test_data <- credit_data[-train_indices, ]
cat("Training data size:", nrow(train_data), "\n")
cat("Testing data size:", nrow(test_data), "\n\n")

#Build the logistic regression model
model <- glm(good_credit ~ V1 + V2 + V3 + V4 + V5 + V6 + V8 + V9 + V10 + V13 + V14 + V16,
             data = train_data,
             family = binomial(link = "logit"))

#show the model results
cat("=== MODEL SUMMARY ===\n")
print(summary(model))

#Make predictions on TEST data
test_data$predicted_prob <- predict(model, newdata = test_data, type = "response")

#Find the BEST threshold
# Let's try different thresholds and calculate total cost
thresholds <- seq(0.3, 0.8, by = 0.05)
costs <- numeric(length(thresholds))

for (i in 1:length(thresholds)) {
  threshold <- thresholds[i]
  predicted_class <- ifelse(test_data$predicted_prob >= threshold, 1, 0)
  
  # False Positive: predicted good (1) but actually bad (0) - COST = 5
  false_positives <- sum(predicted_class == 1 & test_data$good_credit == 0)
  
  # False Negative: predicted bad (0) but actually good (1) - COST = 1
  false_negatives <- sum(predicted_class == 0 & test_data$good_credit == 1)
  
  # Total cost
  costs[i] <- (false_positives * 5) + (false_negatives * 1)
}

# Find best threshold (minimum cost)
best_index <- which.min(costs)
best_threshold <- thresholds[best_index]

plot(thresholds, costs, 
     type = "b", 
     col = "blue", 
     lwd = 2,
     pch = 19,
     xlab = "Threshold Probability",
     ylab = "Total Cost",
     main = "Finding the Best Threshold")
abline(v = best_threshold, col = "red", lwd = 2, lty = 2)
text(best_threshold, max(costs), 
     paste("Best:", best_threshold), 
     pos = 4, col = "red")


cat("\n=== THRESHOLD ANALYSIS ===\n")
result_table <- data.frame(
  Threshold = thresholds,
  Total_Cost = costs
)
print(result_table)

cat("\n*** BEST THRESHOLD:", best_threshold, "***\n")
cat("This minimizes the cost considering bad->good errors are 5x worse\n\n")

#Evaluate model quality with BEST threshold
predicted_class <- ifelse(test_data$predicted_prob >= best_threshold, 1, 0)

boxplot(predicted_prob ~ good_credit, 
        data = test_data,
        col = c("salmon", "lightgreen"),
        names = c("Bad Credit (0)", "Good Credit (1)"),
        ylab = "Predicted Probability",
        main = "Model Predictions by Actual Credit")
abline(h = best_threshold, col = "red", lwd = 2, lty = 2)

# Confusion Matrix
confusion <- table(Predicted = predicted_class, Actual = test_data$good_credit)
cat("=== CONFUSION MATRIX ===\n")
print(confusion)

confusion_pct <- prop.table(confusion, margin = 2) * 100
barplot(confusion_pct,
        beside = TRUE,
        col = c("salmon", "lightgreen"),
        legend = c("Predicted Bad", "Predicted Good"),
        xlab = "Actual Credit Status",
        ylab = "Percentage",
        main = "Prediction Accuracy",
        names.arg = c("Actually Bad", "Actually Good"))

# Calculate metrics
accuracy <- sum(predicted_class == test_data$good_credit) / nrow(test_data)
sensitivity <- confusion[2,2] / sum(confusion[,2])  # True positive rate
specificity <- confusion[1,1] / sum(confusion[,1])  # True negative rate

cat("\n=== MODEL QUALITY ===\n")
cat("Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Sensitivity (catching good customers):", round(sensitivity * 100, 2), "%\n")
cat("Specificity (catching bad customers):", round(specificity * 100, 2), "%\n")

#show what the coefficients mean
cat("\n=== WHAT THE MODEL LEARNED (Simplified) ===\n")
cat("Positive coefficient = increases chance of being good credit\n")
cat("Negative coefficient = decreases chance of being good credit\n")
cat("Bigger number = stronger effect\n\n")

coefs <- summary(model)$coefficients[,1]
cat("Top factors:\n")
print(round(coefs, 3))






