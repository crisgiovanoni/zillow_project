import seaborn as sns
from sklearn.metrics import mean_squared_error, explained_variance_score
from math import sqrt

def plot_residuals(x, y, dataframe):
    """
    Takes the feature, the target, and the dataframe as input, and
    Returns a residual plot.
    """
    return sns.residplot(x,y,data=dataframe)

def regression_errors(y, yhat):
    """
    Takes in y and yhat, then
    Returns the sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), mean squared error (MSE) and root mean squared error (RMSE).
    """
    mse = mean_squared_error(y,yhat)
    sse = mse * len(y)
    ess = ((yhat - y.mean())**2).sum()
    rmse = sqrt(mse)
    tss = ess + sse
    return sse, mse, rmse, ess, tss

def baseline_mean_errors(y):
    """
    Takes in your target, y, computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and
    Returns the error values (SSE, MSE, and RMSE).
    """
    df = pd.DataFrame()
    df["y"] = y
    df["yhat"] = y.mean()
    sse_baseline = (mean_squared_error(df.y, df.yhat)) * len(df.y)
    mse_baseline = mean_squared_error(y,yhat)
    rmse_baseline = sqrt(mse_baseline)
    return sse_baseline, mse_baseline, rmse_baseline

def better_than_baseline(ess, sse_baseline,ols_model):
    """
    Returns true if your model performs better than the baseline, otherwise false.
    """
    r2 = ess/sse_baseline
    f_pval = ols_model.f_pvalue
    f_pval < 0.05 = True

def model_significance(ols_model):
    """
    Takes the ols model as input, and
    Returns the amount of variance explained in your model, and the value telling you whether the correlation between the model and the y are statistically significant.
    """
    r2 = ols_model.rsquared
    return r2, ols.model.f_pvalue, ols_model