import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np.random.seed(1114)

if __name__ == "__main__":
# Load data
    print("Loading data...", flush=True)
    df = pd.read_csv("../US_Accidents_Dec21_updated.csv")
    df = df[df.select_dtypes('number').columns]
    df = df.drop(["End_Lat", "End_Lng", "Distance(mi)"], axis=1)
    df = df.dropna()
    
# OLS Analysis
    X_train, X_test, y_train, y_test = train_test_split(df.drop("Severity", axis=1), df["Severity"], test_size=0.2)
    
    model = sm.OLS(y_train, X_train).fit()
    #print(model.summary()) # Adj R-squared = 0.967
    
    X_train = X_train.drop(["Precipitation(in)"], axis=1)
    X_test = X_test.drop(["Precipitation(in)"], axis=1)
    model = sm.OLS(y_train, X_train).fit()
    #print(model.summary()) # Adj R-squared = 0.967
    prediction = model.predict(X_test)
    MSE = np.square(np.subtract(y_test, prediction)).mean()
    print(f'MSE using OLS is {np.around(MSE, 3)}') # 0.143
    
    residuals = y_test - prediction
    sm.qqplot(residuals, line='45', fit=True) 
    plt.title('QQ Plot')
    plt.savefig('reg.png')
    plt.show()
    
# Colinearity anaysis
    X = df.drop(["Severity", "Precipitation(in)"], axis=1)
    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info['Column'] = X.columns
    vif_info = vif_info.sort_values('VIF', ascending=False)
    vif_info = vif_info.round(3)
    print(vif_info)

# Confident interval
    print(np.around(model.conf_int(0.05), 3))
    predictions = model.get_prediction(X_test)
    summary = predictions.summary_frame(alpha=0.05)
    print(summary)
