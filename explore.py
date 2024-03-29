# Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise relationships along with the regression line for each pair.
# Write a function, months_to_years(tenure_months, df) that returns your dataframe with a new feature tenure_years, in complete years as a customer.
# Write a function, plot_categorical_and_continous_vars(categorical_var, continuous_var, df), that outputs 3 different plots for plotting a categorical variable with a continuous variable, e.g. tenure_years with total_charges. For ideas on effective ways to visualize categorical with continuous: https://datavizcatalogue.com/. You can then look into seaborn and matplotlib documentation for ways to create plots.

import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable_pairs(dataframe):