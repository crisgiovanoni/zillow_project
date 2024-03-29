# Standard
import pandas as pd
import numpy as np
import env

# Split-scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, MinMaxScaler, RobustScaler

# Model
from sklearn.linear_model import LinearRegression

# =========== WORKING ZILLOW FUNCTIONS ============== #

def get_db_url(db):
    """
    Produces a url from env credentials
    >> Input:
    database
    << Output:
    url
    """
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

#######!!!!!!!!! check your selection on usetype
def get_sql_zillow():
    """
    Queries from zillow database with the following conditions:
    - Single-unit properties
    - 
    >> Input:
    database
    << Output:
    url
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
    # Filter Transaction Dates by boolean mask and loc
    df.transactiondate = pd.to_datetime(df.transactiondate, format='%Y-%m-%d')
    filter_dates = (df.transactiondate >= '2017-05-01') & (df.transactiondate <= '2017-06-30') 
    df = df.loc[filter_dates]
    # Rename columns
    df = df.rename(columns={"calculatedfinishedsquarefeet": "sqft","bathroomcnt":"bathcnt","bedroomcnt":"bedcnt","transactiondate":"transdate","taxvaluedollarcnt":"propvalue","parcelid":"id"})
    # Drop all NaNs
    df = df.dropna()
    return df

def split_my_data(X, y, train_ratio=0.7):
    # Use when X and y data frames are available, and split train and test for modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=123)
    return X_train, X_test, y_train, y_test

def standard_scaler(X_train, X_test):
    standard_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    # Scale Train Data and Convert to a Data Frame
    scaled_X_train = standard_scaler.transform(X_train)
    scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns).set_index([X_train.index])
    # Scale Train and Convert to a Data Frame
    scaled_X_test = standard_scaler.transform(X_test)
    scaled_X_test = pd.DataFrame(X_test, columns=X_test.columns).set_index([X_test.index])
    return scaled_X_train, scaled_X_test, standard_scaler

def generate_linear_model(scaled_X_df, y_df):
    lm = LinearRegression()
    lm.fit(scaled_X_df, y_df)
    lm_intercept = lm.intercept_
    lm_coefficients = lm.coef_
    return lm, lm_intercept, lm_coefficients

def predict_on_test(lm, scaled_X_test):
    yhat = lm.predict(scaled_X_test)
    return yhat

# =========== TESTING ZILLOW FUNCTIONS ============== #

train_ratio = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=123)
len(X_train) #10998
len(X_test) #4714
len(y_train) #10998
len(y_test) #4714

lm, lm_intercept, lm_coefficients = generate_linear_model(X_train,y_train)
lm
lm_intercept
lm_coefficients

yhat

# =================================================== 


