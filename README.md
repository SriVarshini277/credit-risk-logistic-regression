# Credit Risk Classification Using Logistic Regression
## Project Overview
This project builds a logistic regression model to predict whether a credit applicant is a good or bad credit risk using the German Credit dataset from the UCI Machine Learning Repository. The model incorporates a cost-sensitive decision threshold, reflecting the higher cost of approving bad credit applicants compared to rejecting good ones.

## Dataset
- **Source:** UCI Machine Learning Repository – German Credit Data
- **Observations:** 1000
- **Features:** 20 input variables
- **Target Variable:** Credit risk (Good vs Bad)

## Methodology
- Split the dataset into **70% training** and **30% testing**
- Built a logistic regression model using selected predictors to avoid overfitting
- Used `glm()` in R with a binomial (logit) link function
- Evaluated multiple probability thresholds to reflect asymmetric misclassification costs

Thresholds between **0.30 and 0.80** were evaluated, and the threshold minimizing total cost was selected.

## Results
- **Optimal Probability Threshold:** 0.80
- **Accuracy:** 67.67%
- **Sensitivity (Good Customers):** 62.25%
- **Specificity (Bad Customers):** 79.17%

The selected threshold prioritizes identifying bad credit risks, aligning with the asymmetric cost structure.

## Key Takeaways
- Cost-sensitive classification significantly impacts model decision behavior
- Higher specificity was achieved by sacrificing some sensitivity
- Logistic regression provides an interpretable and effective baseline for credit risk modeling

## Tools & Technologies
- R
- Logistic Regression (`glm`)
- Confusion Matrix & Classification Metrics
- Data Visualization

## How to Run
1. Download the dataset from the UCI repository
2. Place the dataset in the project directory
3. Run the R script in RStudio or via command line
