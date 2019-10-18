import pandas as pd
import numpy as np

import env

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_sql_zillow():
    query = '''
    SELECT prop.parcelid, calculatedfinishedsquarefeet, bathroomcnt, bedroomcnt, transactiondate, taxvaluedollarcnt
    FROM properties_2017 as prop
    JOIN predictions_2017 as pred
	    USING(parcelid)
    JOIN propertylandusetype as usetype
	    ON prop.propertylandusetypeid = usetype.propertylandusetypeid
	    AND prop.propertylandusetypeid in (261,262,263,264,265,268,269,273,275,276)
    ORDER BY transactiondate
    ''' 
    df = pd.read_sql(query, get_db_url("zillow"))
    return df

zillow = get_sql_zillow()
len(zillow)
zillow.head(1000)
zillow.dtypes

zillow.sort_values(by=zillow.transactiondate)

# Filter Transaction Dates by boolean mask and loc
zillow.transactiondate = pd.to_datetime(zillow.transactiondate, format='%Y-%m-%d')
filter_dates = (zillow.transactiondate >= '2017-05-01') & (zillow.transactiondate <= '2017-06-30') 
zillow = zillow.loc[filter_dates]
# Rename columns
zillow.rename(columns={
    "calculatedfinishedsquarefeet": "sqft","bathroomcnt":"bathcnt","bedroomcnt":"bedcnt","transactiondate":"transdate","taxvaluedollarcnt":"propvalue"})

zill

def filter_zillow():
    df = df[transactiondate =]

# df = get_data_from_mysql()
# df.head()

# def clean_data(df):
#     df = df[['customer_id', 'total_charges', 'monthly_charges', 'tenure']]
#     df.total_charges = df.total_charges.str.strip().replace('', np.nan).astype(float)
#     df = df.dropna()
#     return df

def clean_data(df):
    df = df[df.total_charges != ' ']
    df.total_charges = df.total_charges.astype(float)
    return df

# df = clean_data(df)
# df.head()
    
def wrangle_telco():
    return clean_data(get_data_from_mysql())

def wrangle_zillow():
    return clean_data(get_sql_zillow())


# data = wrangle_telco()
# data.head()