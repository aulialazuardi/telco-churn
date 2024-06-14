# Telco Customer Churn Prediction: Oversampling & Undersampling
## Background
Customer churn in telco industry is a huge issue, since customer acquisition is quite a huge burden in terms of resource and marketing for telco operators. 
Furthermore, customer retention by offering personalized reward and/or experience is much cheaper than initial investment in customer acquisition previously mentioned. 
This project aims to present an analysis and provide a model that predicts customer who about to churn, in hope it will help the customer retention.

The dataset used in this project is obtained from: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
In itself has 21 Features, including the Target column, namely Churn. To provide an organized analysis, the EDA of the features are separated into:
- Demographics (Gender, SeniorCitizens,Partnership, Dependants)
- Services Subscribed ( MultipleLInes,Techsupport, StreamingTV, Streaming Movies...)
- Billing & Tenure ( Contract, PaperlessBilling, PaymentMethod...)

## Data Cleaning
- Check all columns types, if the column supposed to be numerical but listed as object, applied pd.to_numeric()
- Eight whitespaces values found, removed since all classified as "No" in Churn column since the dataset are imbalanced towards "No"
- Check duplicates and no duplicates found

## Exploratory Data Analysis
1. Demographics Category were plotted in respect to Churn Column, all were categorical features.
2. Services Subscribed also plotted in respect to Churn Column, all were also categorical features. 6 Features were joined as a new feature to see the total of services subscribed.
3. Billing & Tenure were numerical & Categorical column, the numerical column were plotted using histogram to see its distribution.

## Preprocessing & Feature Engineering
### Correlated Features - New Feature
Features were plotted to see its correlation in respect to each other. Features with high correlation were the 6 features previously mentioned in EDA, the highest correlation to Churn were kept while others were removed.
The new feature created in EDA will be used in lieu of the highly correlated features.
### Feature Encoding & One Hot Encoding
All the categorical columns were treated with label encoding but one (PaymentMethod). Since PaymentMethod is not ordinal, One Hot Encoding were used.
### Standardizing Numerical Features
The numerical features were scaled & standardized after splitting the test & train set (30% test data) to avoid data leakage.

## Model Comparison
Three classifier models were tested as the baseline model before any methods were introduced, the models are:
- Logistic Regression
- Random Forest
- XGBoost Model
All the models were compared by the Precision-Recall Curve and the Recall score. Nodel's hyperparameter is not touched. XGBoost were the lowest and won't be compared in the next subsection
## Oversampling & Undersampling
Since it is clear from the EDA that the dataset has class imbalance, Methods to equalize the class were introduced. The methods are Naive Oversampling & Naive Undersampling
### Oversampling
- Minority Class were oversampled and tested between two models (Logistic Regression & Random Forest)
- Huge Improvement in Recall Score, but Area Under P-R Curve slightly different from Baseline Model, due to huge drop in Precision
### Undersampling
- Majority Class were removed until equal with the minority class
- Recall Score were slightly better than Oversampled model for both Models
### Combination of both
- Majority class were removed and minority class were oversampled until reaches the middle ground.
- Recal, Precision and P-R Curve were suffering and lower than Undersampling, but higher than oversampling.

## Conclusion
Undersampled Logistic Regression gives the best Recall & P-R Curve score. to improve other models, weighted Random Forest and HyperParameter Tuning is highly encouraged. Other sampling methods (SMOTE, ADASYN, etc) perhaps could improve the Recall score as well.
