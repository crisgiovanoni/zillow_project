# Standard
import pandas as pd
import numpy as np
import env
import seaborn as sns

from pandas import ExcelWriter
from pandas import ExcelFile

# Split-scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, MinMaxScaler, RobustScaler, Normalizer

# Model
from sklearn.linear_model import LinearRegression

#### ACQUIRE

fips = pd.read_excel('US_FIPS_Codes.xls', sheetname='3,142 U.S. Counties')

print("Column headings:")
print(df.columns)

def get_db_url(db):
    """
    Produces a url from env credentials
    >> Input:
    database
    << Output:
    url
    """
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_sql_zillow():
    """
    Queries from zillow database with the following specs:
    - Fields: Parcel ID, Calculated Finished Square Footage, No. of Bathrooms, No. of Bedrooms, Last Transaction Date, Property Value
    >> Input:
    none
    << Output:
    data frame
    """
    query = '''
    SELECT prop.parcelid, calculatedfinishedsquarefeet, bathroomcnt, bedroomcnt, transactiondate, taxvaluedollarcnt
    FROM properties_2017 as prop
    JOIN predictions_2017 as pred
	    USING(parcelid)
    JOIN propertylandusetype as usetype
	    ON prop.propertylandusetypeid = usetype.propertylandusetypeid
	    AND prop.propertylandusetypeid in (261,262,263,264,265,268,269,273,275,276)
    ''' 
    df = pd.read_sql(query, get_db_url("zillow"))
    return df

def filter_zillow_baseline(df):
    """
    Filters zillow query for baseline model with the following conditions: 
    - Change date data type from string to date
    - Renames queried fields
    - Removes NaN
    >> Input:
    data frame
    << Output:
    filtered data frame
    """
    # Filter Transaction Dates by boolean mask and loc
    df.transactiondate = pd.to_datetime(df.transactiondate, format='%Y-%m-%d')
    filter_dates = (df.transactiondate >= '2017-05-01') & (df.transactiondate <= '2017-06-30') 
    df = df.loc[filter_dates]
    # Rename columns
    df = df.rename(columns={"calculatedfinishedsquarefeet": "sqft","bathroomcnt":"bathcnt","bedroomcnt":"bedcnt","transactiondate":"transdate","taxvaluedollarcnt":"propvalue","parcelid":"id"})
    # Drop all NaNs
    df = df.dropna()
    return df

def filter_zillow(df):
    """
    Filters zillow query for baseline model with the following conditions: 
    - Changes date data type from string to date
    - Renames queried fields
    >> Input:
    data frame
    << Output:
    filtered data frame
    """
    # Filter Transaction Dates by boolean mask and loc
    df.transactiondate = pd.to_datetime(df.transactiondate, format='%Y-%m-%d')
    filter_dates = (df.transactiondate >= '2017-05-01') & (df.transactiondate <= '2017-06-30') 
    df = df.loc[filter_dates]
    # Rename columns
    df = df.rename(columns={"calculatedfinishedsquarefeet": "sqft","bathroomcnt":"bathcnt","bedroomcnt":"bedcnt","transactiondate":"transdate","taxvaluedollarcnt":"propvalue","parcelid":"id"})
    return df

#### PRE-PROCESS

def split_my_data(X, y, train_ratio=0.7):
    """
    Queries from zillow database with the following specs:
    - Fields: Parcel ID, Calculated Finished Square Footage, No. of Bathrooms, No. of Bedrooms, Last Transaction Date, Property Value
    >> Input:
    Data frame with X
    Data frame with y
    Ratio of resulting train data
    << Output:
    X_train
    X_test
    y_train
    y_test
    """
    # Use when X and y data frames are available, and split train and test for modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=123)
    return X_train, X_test, y_train, y_test

def standard_scaler(X_train, X_test):
    """
    Scales X_train and X_test using Standard Scaler
    >> Input:
    X_train
    X_test
    << Output:
    scaled_X_train
    scaled_X_test
    standard_scaler object
    """
    standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    # Scale Train Data and Convert to a Data Frame
    scaled_X_train = standard_scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns).set_index([X_train.index])
    # Scale Train and Convert to a Data Frame
    scaled_X_test = standard_scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns).set_index([X_test.index])
    return scaled_X_train, scaled_X_test, standard_scaler

def gaussian_scaler(X_train, X_test):
    # Creates a Gaussian Scaler object and fit Train Data 
    gaussian_scaler = PowerTransformer(method="yeo-johnson", standardize=False, copy=True).fit(X_train)
    # Scale Train Data and Convert to a Data Frame
    scaled_X_train = gaussian_scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns.values).set_index([X_train.index.values])
    # Scale Train and Convert to a Data Frame
    scaled_X_test = gaussian_scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns.values).set_index([X_test.index.values])
    return scaled_X_train, scaled_X_test, gaussian_scaler

def normalize_scaler(X_train, X_test):
    scaler = Normalizer().fit(X_train)

    scaled_X_train = scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns).set_index([X_train.index])
    
    scaled_X_test = scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns).set_index([X_test.index])
    return scaled_X_train, scaled_X_test, scaler

def robust_scaler(X_train, X_test):
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True).fit(X_train)

    scaled_X_train = scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns).set_index([X_train.index])
    
    scaled_X_test = scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns).set_index([X_test.index])
    return scaled_X_train, scaled_X_test, scaler

def quantile_scaler(X_train, X_test):
    scaler = QuantileTransformer(n_quantiles=1000,output_distribution='normal',random_state=123,copy=True).fit(X_train)

    scaled_X_train = scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns).set_index([X_train.index])
    
    scaled_X_test = scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns).set_index([X_test.index])
    return scaled_X_train, scaled_X_test, scaler

def generate_linear_model(scaled_X_df, y_df):
    """
    Generates linear model.
    Use scaled X data and unscaled y data.
    >> Input:
    scaled_X_df
    y_df
    << Output:
    linear model
    """
    lm = LinearRegression()
    lm.fit(scaled_X_df, y_df)
    lm.predict(scaled_X_df)
    lm_intercept = lm.intercept_
    lm_coefficients = lm.coef_
    return lm, lm_intercept, lm_coefficients

def plot_residuals(X, y, df):
    '''
    Plots the residuals of a linear model.
    >> Input:
    X
    y
    << Output:
    seaborn plot
    '''
    return sns.residplot(X, y,df,lowess=True)
# # =========== TESTING ZILLOW FUNCTIONS ============== #

# train_ratio = 0.7
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=123)
# len(X_train) #10998
# len(X_test) #4714
# len(y_train) #10998
# len(y_test) #4714

# lm, lm_intercept, lm_coefficients = generate_linear_model(X_train,y_train)
# lm
# lm_intercept
# lm_coefficients

# yhat
# # =================================================== 


