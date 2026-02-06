# 24ADI003_24BAD122
Scenario 1: Ocean Water Temperature Prediction (Linear Regression)

Dataset (Kaggle – Public): 
https://www.kaggle.com/datasets/sohier/calcofi

In Scenario 1, Linear Regression is used to predict ocean water temperature using environmental and depth-related features. The CalCOFI dataset obtained from Kaggle is used for this purpose. Water temperature (T_degC) is taken as the target variable, while depth, salinity, oxygen, latitude, and longitude are used as input features. The dataset is preprocessed by handling missing values using mean or median imputation and applying feature scaling to improve model performance. The data is then split into training and testing sets, and a Linear Regression model is trained. Model performance is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score. Further improvements are achieved using feature selection and regularization techniques such as Ridge and Lasso regression to reduce overfitting and enhance prediction accuracy.

Scenario 2: LIC Stock Price Movement Prediction (Logistic Regression)

Dataset (kaggle-public)
https://www.kaggle.com/datasets/debashis74017/lic-stock-price-data

In Scenario 2, Logistic Regression is applied to classify whether the LIC stock price will increase or decrease based on historical stock market data. The LIC stock dataset from Kaggle is used, and a binary target variable called price movement is created, where a value of 1 indicates that the closing price is higher than the opening price and 0 indicates otherwise. The input features include open price, high price, low price, and trading volume. The dataset is preprocessed by handling missing values and performing feature scaling to ensure faster convergence. The data is split into training and testing sets, and a Logistic Regression model is trained. The model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix. ROC curve analysis and hyperparameter tuning with regularization are performed to improve classification performance and reduce overfitting.
