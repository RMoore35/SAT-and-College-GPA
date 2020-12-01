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

# Running a basic linear regression, therefore just one x variable, SAT
df = pd.DataFrame(mydata, columns=['SAT', 'COLGPA'])
display(df.head(n=5))
X = df[['SAT']]
Y = df['COLGPA']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())
print('R-Squared: \n', results.rsquared)
print('Adjusted R-squared: \n', results.rsquared_adj)
print('Standard errors of coefficients: \n', results.bse)
print('Standard error or regression: \n', np.sqrt(results.scale))
