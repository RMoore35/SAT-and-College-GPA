import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display

# Read in data from local excel file
mydata = pd.read_excel(
    '/Users/ryan/Google Drive/Tulane/Job Stuff/Portfolio Projects/SAT-and-College-GPA/data for jan 27th.xlsx', sheet_name='my_data')

df = pd.DataFrame(mydata, columns=['SAT', 'female', 'athlete', 'COLGPA'])
display(df.head(n=5))
X = df[['SAT', 'female', 'athlete']]
Y = df['COLGPA']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
