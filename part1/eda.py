import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1114)

if __name__ == "__main__":
    df = pd.read_csv("../US_Accidents_Dec21_updated.csv")
    '''print(df.shape) # (2845342, 47)
    print("Number of categorical features: ", len(df.select_dtypes(exclude='number').columns)) # 33
    print("Number of numerical features: ", len(df.select_dtypes('number').columns)) # 14'''
    #print(df.head())
    sns.set_theme(style="whitegrid")

# Severity
    '''plt.figure()
    graph = sns.countplot(x='Severity', data=df, palette="Set2")
    graph.bar_label(graph.containers[0])
    plt.title("Accident counts in each severity")
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.savefig('accident_severity.png')
    plt.show()'''

# Duration
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time']) 
    df['Duration'] = df.End_Time - df.Start_Time 
    df['Duration'] = df['Duration'].apply(lambda x: x.total_seconds() / 60)
    print("The overall mean duration is: {} minutes".format(np.around(df['Duration'].mean(), 2))) # 359.03

    '''plt.figure()
    sns.boxplot(x="Severity", y="Duration", data=df.loc[df['Duration'] < 400])
    plt.title('Duration on each severity level')
    plt.xlabel('Severity')
    plt.ylabel('Duration(mins)')
    plt.savefig('severe_duration.png')
    plt.show()'''
    
# State
    '''unique_states = df['State'].unique()
    state_cnt=[]
    for i in unique_states:
        state_cnt.append(df[df['State'] == i].count()['ID'])
        if i == 'CA': # 795868
            print('CA', df[df['State'] == i].count()['ID'])
        if i == 'FL': # 401388
            print('FL', df[df['State'] == i].count()['ID'])
        if i == 'VA': # 113535
            print('VA', df[df['State'] == i].count()['ID'])
        if i == 'OR': # 126341
            print('OR', df[df['State'] == i].count()['ID'])
        if i == 'TX': # 149037
            print('TX', df[df['State'] == i].count()['ID'])
        if i == 'NY': # 108049
            print('NY', df[df['State'] == i].count()['ID'])
            
    plt.figure(figsize=(18, 12))
    sns.barplot(x=unique_states, y=state_cnt)
    #plt.bar(states, count_by_state)
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.title('Accident counts in each state', size=25)
    plt.savefig('accident_state.png')
    plt.show()'''

# Timezone
    '''plt.figure()
    sns.countplot(x='Timezone', data=df, palette="Set2")
    plt.title("Accident counts in each timezone")
    plt.xlabel('Timezone')
    plt.ylabel('Count')
    plt.savefig('accident_timezone.png')
    plt.show()'''

# Drop useless features (TODO: need more analysis)
    df = df.drop(["ID", "Description", "Distance(mi)", "End_Time", "End_Lat", "End_Lng"], axis=1)
    #print("The shape of data is:",(df.shape))

# Drop categorical features with only one unique value (TODO: more categorical?)
    df = df.drop(["Country", "Turning_Loop"], axis=1) 
    #print("The shape of data is:",(df.shape))


# Missing Value
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name','missing_count']
    missing_df = missing_df.loc[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by='missing_count')

    ind = np.arange(missing_df.shape[0])
    width = 0.5
    '''plt.figure(figsize=(16, 24))
    plt.barh(ind, missing_df.missing_count.values, color='blue')
    #plt.yticks(ind)
    #plt.yticklabels(missing_df.column_name.values, rotation='horizontal')
    plt.yticks(ticks=np.arange(missing_df.shape[0]), labels=missing_df.column_name.values)
    plt.xlabel("Count")
    plt.title("Number of missing values in each column", size=25)
    plt.grid('x')
    plt.savefig("missing_value.png")
    plt.show() # drop Number and Wind_CHill(F)'''


# Weather condition
    df['Weather_Condition'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), 'Cloud', df.Weather_Condition)
    df['Weather_Condition'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), 'Rain', df.Weather_Condition)
    df['Weather_Condition'] = np.where(df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), 'Heavy_Rain', df.Weather_Condition)
    df['Weather_Condition'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), 'Snow', df.Weather_Condition)
    df['Weather_Condition'] = np.where(df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), 'Heavy_Snow', df.Weather_Condition)
    df['Weather_Condition'] = df['Weather_Condition'].loc[(df['Weather_Condition'] == 'Clear') |
                                                        (df['Weather_Condition'] == 'Cloud') |
                                                        (df['Weather_Condition'] == 'Rain') |
                                                        (df['Weather_Condition'] == 'Heavy_Rain') |
                                                        (df['Weather_Condition'] == 'Snow') |
                                                        (df['Weather_Condition'] == 'Heavy_Snow') |
                                                        (df['Weather_Condition'] == 'Fog')]
    '''plt.figure(figsize=(16, 12))
    df['Weather_Condition'].value_counts().sort_values(ascending=False).plot.bar()
    plt.xlabel('Weather Condition')
    plt.ylabel('Count')
    plt.title('Accident counts in each weather condition', size=25)
    plt.savefig("accident_weather.png")
    plt.show()'''

# Simplify Wind directions
    #print(df['Wind_Direction'].unique())
    df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
    df.loc[(df['Wind_Direction']=='West')|(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
    df.loc[(df['Wind_Direction']=='South')|(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
    df.loc[(df['Wind_Direction']=='North')|(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
    df.loc[(df['Wind_Direction']=='East')|(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
    df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
    #print("Wind Direction after simplification: ", df['Wind_Direction'].unique()) # ['SW' 'CALM' 'W' 'N' 'S' 'NW' 'E' 'SE' nan 'VAR' 'NE']
    # Wind directions
    '''plt.figure(figsize=(12, 8))
    sns.countplot(x='Wind_Direction', data=df, palette="Set2")
    plt.title("Accident counts in each wind Direction")
    plt.xlabel('Wind Condition')
    plt.ylabel('Count')
    plt.savefig("accident_wind.png")
    plt.show()'''


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
    missing = missing.round(3)
    missing = missing.sort_values(by="Missing_Percent(%)", ascending=False)
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

# Correlation matrix   
    '''fig = plt.gcf()
    fig.set_size_inches(20,20)
    fig= sns.heatmap(df.drop("Severity", axis=1).corr(),annot=True,linewidths=1,linecolor='k',
                    square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
    plt.title('Correlation Matrix', size=25)
    plt.savefig('correlation.png')
    plt.show()'''

# Feature Importance
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    df = df[df.select_dtypes('number').columns]
    print(df.shape)
    X_train, X_test, y_train, y_test = train_test_split(df.drop("Severity", axis=1), df["Severity"], test_size=0.2)
    
    rf = RandomForestRegressor(random_state=1)
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()
    plt.figure(figsize=(18, 12))
    plt.barh(df.drop("Severity", axis=1).columns[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Label")
    plt.title("Random Forest Feature Importance")
    plt.grid('x')
    plt.grid('y')
    plt.savefig("feature_importance.png")
    plt.show()

# PCA
    print(df.shape)
    print("Number of categorical features: ", len(df.select_dtypes(exclude='number').columns))
    print("Number of numerical features: ", len(df.select_dtypes('number').columns))
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2, svd_solver='full')
    pca.fit(X_train)
    X_PCA = pca.transform(X_train)
    print("Original Dim", X_train.shape)
    print("Transformed Dim", X_PCA.shape)
    PCA(n_components='mle',svd_solver='full')
    print(f'explained variance ratio {pca.explained_variance_ratio_}')
    print(f'singular values of transformed datam{pca.singular_values_}')
    plt.figure()
    plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1),np.cumsum(pca.explained_variance_ratio_))
    plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
    plt.grid()
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('pca.png')
    plt.show()