import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1114)

if __name__ == "__main__":
    df = pd.read_csv("../US_Accidents_Dec21_updated.csv")
    print(df.columns)
    
# Drop useless features (TODO: need more analysis)
    df = df.drop(["ID", "Description", "Distance(mi)", "End_Time",
                  "End_Lat", "End_Lng"], axis=1)
    #print("The shape of data is:",(df.shape))

# Drop categorical features with only one unique value (TODO: more categorical?)
    df = df.drop(["Country", "Turning_Loop"], axis=1) 
    #print("The shape of data is:",(df.shape))
    
# Normalize Wind Direction
    print("Original Wind Directions: ", df["Wind_Direction"].unique())
    df.loc[df["Wind_Direction"] == "Calm", "Wind_Direction"] = "CALM"
    df.loc[(df["Wind_Direction"] == "West") | (df["Wind_Direction"] == "WSW") | (df["Wind_Direction"] == "WNW"), "Wind_Direction"] = 'W'
    df.loc[(df["Wind_Direction"] == "South") | (df["Wind_Direction"] == "SSW") | (df["Wind_Direction"] == "SSE"), "Wind_Direction"] = 'S'
    df.loc[(df["Wind_Direction"] == "North") | (df["Wind_Direction"] == "NNW") | (df["Wind_Direction"] == "NNE"), "Wind_Direction"] = 'N'
    df.loc[(df["Wind_Direction"] == "East") | (df["Wind_Direction"] == "ESE") | (df["Wind_Direction"] == "ENE"), "Wind_Direction"] = 'E'
    df.loc[df["Wind_Direction"] == "Variable", "Wind_Direction"] = "VAR"
    print("Wind Directions after normalization: ", df["Wind_Direction"].unique())
    print("The shape of data is:",(df.shape))

# Normalize Weather Condition
    df["Clear"] = np.where(df["Weather_Condition"].str.contains("Clear", case=False, na=False), True, False)
    df["Cloud"] = np.where(df["Weather_Condition"].str.contains("Cloud|Overcast", case=False, na=False), True, False)
    df["Rain"] = np.where(df["Weather_Condition"].str.contains("Rain|storm", case=False, na=False), True, False)
    df["Heavy_Rain"] = np.where(df["Weather_Condition"].str.contains("Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms", case=False, na=False), True, False)
    df["Snow"] = np.where(df["Weather_Condition"].str.contains("Snow|Sleet|Ice", case=False, na=False), True, False)
    df["Heavy_Snow"] = np.where(df["Weather_Condition"].str.contains("Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls", case=False, na=False), True, False)
    df["Fog"] = np.where(df["Weather_Condition"].str.contains("Fog", case=False, na=False), True, False)
    weather = ["Clear", "Cloud", "Rain", "Heavy_Rain", "Snow", "Heavy_Snow", "Fog"]
    # Handle missing weathers
    mask = df["Weather_Condition"].isnull()
    for w in weather:
        df.loc[mask, w] = df.loc[mask, "Weather_Condition"]
        df[w] = df[w].astype("bool")
    df = df.drop(["Weather_Condition"], axis=1)
    print("The shape of data is:",(df.shape))

# Fix Datetime format
    start_time = pd.DatetimeIndex(df["Start_Time"])
    df["Year"] = start_time.year
    nmonth = start_time.month
    df["Month"] = nmonth
    df["Weekday"]= start_time.weekday
    days_each_month = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
    nday = [days_each_month[arg - 1] for arg in nmonth.values]
    nday = nday + start_time.day.values
    df["Day"] = nday
    df["Hour"] = start_time.hour
    df["Minute"] = df["Hour"] * 60.0 + start_time.minute
    df = df.drop(["Weather_Timestamp", "Start_Time"], axis=1)
    print("The shape of data is:",(df.shape))
    
# Handle missing values
    missing = pd.DataFrame(df.isnull().sum()).reset_index()
    missing.columns = ["Feature", "Missing_Percent(%)"]
    missing["Missing_Percent(%)"] = missing["Missing_Percent(%)"].apply(lambda x: x / df.shape[0] * 100)
    print(missing.loc[missing["Missing_Percent(%)"] > 0, :])

    # Drop features
    df = df.drop(["Number", "Wind_Chill(F)"], axis=1)
    
    # Flag NaN data and fill with median
    df["Precipitation_NA"] = 0
    df.loc[df["Precipitation(in)"].isnull(), "Precipitation_NA"] = 1
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(df["Precipitation(in)"].median())

    # Drop NaN data
    df = df.dropna()
    print(df.shape)
    
# Transform categorical features to binary encoding
    obj_cols = df.select_dtypes(include="object").columns
    binary_encoder = ce.BinaryEncoder(cols=obj_cols)
    df = binary_encoder.fit_transform(df)
    # TODO: Use one-hot encoding with condensation of minorities
    print(df.shape)

# Under sampling severity 2 for label balancing
    s2_mask = (df["Severity"] == 2)
    s2_df = df[s2_mask].sample(int(s2_mask.sum() * 0.1))
    ns2_df = df[~s2_mask]
    df = pd.concat([s2_df, ns2_df])
    print(np.unique(df["Severity"], return_counts=True))

# Split train and test data
    X = df.drop("Severity", axis=1)
    y= df["Severity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Standardization
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)
    X_test = standardizer.fit_transform(X_test)

# Save to disk
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)