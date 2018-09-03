import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None)
df.head(5)
df.tail(5)

##Defining headers for the DataSet
headers=["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels",
        "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
         "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
         "highway-mpg", "price"]

df.columns = headers

#print(df.dtypes)

##Exporting to csv
'''

path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)

'''

##Generating descriptive stats
"""

#print(df.describe(include="all"))
#print(df.info)

"""

##Data Formatting, replacing ? to NAN
"""

df["price"].replace('?',np.nan, inplace = True)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)

"""

##Data Formatting, converting prices from Object to Int, dropping NaN values


df["price"].replace('?',np.nan, inplace = True)
df.dropna(subset=["price"], axis=0, inplace=True)
df["price"] = df["price"].astype("int")

##Data Formatting, converting peak-rpm from Object to Int, dropping NaN values

df["peak-rpm"].replace('?',np.nan, inplace = True)
df.dropna(subset=["peak-rpm"], axis=0, inplace=True)
df["peak-rpm"] = df["peak-rpm"].astype("int")

#print(df.info)


###Data Binning
'''

binwidth = int((max(df["price"])-min(df["price"]))/3)
bins = range(min(df["price"]), max(df["price"]), binwidth)
group_names = ['low','medium','high']
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)
df.dropna(subset=["price-binned"], axis=0, inplace=True)

'''

##Plotting Histogram from the binned value

'''
plt.hist(df["price"],bins=3)
plt.title("Price Bins")
plt.xlabel("Count")
plt.ylabel("Price")
plt.show()
'''


#TURNING CATEGORICAL VARIABLES INTO QUANTITATIVE VARIABLES
'''

df = (pd.get_dummies(df["fuel-type"]))

'''

#DESCRIPTIVE STATISTICS- Value_counts
'''

drive_wheels_counts = df["drive-wheels"].value_counts()
drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

'''

#Box Plots

'''
sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()



#Scatterplot


y=df["engine-size"]
x=df["price"]
plt.scatter(x,y)

plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()



#Group by to visualize price based on drive-wheels and body style.

df_test = df[["drive-wheels", "body-style", "price"]]
df_group = df_test.groupby(['drive-wheels', 'body-style'], as_index = False).mean()

#Pivot Table to visualize price based on drive-wheels and body style.

df_pivot = df_group.pivot(index = 'drive-wheels', columns= 'body-style')
print(df_pivot)


#Heat Maps

plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.show()



#CORRELATION, Positive Linear Relationship between engine size and price


sns.regplot(x='engine-size', y='price', data=df)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.ylim(0,)
plt.show()



#CORRELATION, Negetive Linear Relationship between highway-mpg and price


sns.regplot(x='highway-mpg', y='price', data=df)
plt.title("Scatterplot of highway-mpg vs price")
plt.xlabel("highway-mpg")
plt.ylabel("price")
plt.ylim(0,)
plt.show()



# WEAK CORRELATION between peak-rpm and price


sns.regplot(x='peak-rpm', y='price', data=df)
plt.title("Scatterplot of peak-rpm vs price")
plt.xlabel("peak-rpm")
plt.ylabel("price")
plt.ylim(0,)
plt.show()

'''

# Simple Linear Model Estimator with Distribution plot
'''

lm = LinearRegression()
X=df[["highway-mpg"]]
Y=df["price"]
lm.fit(X,Y)
Yhat1 = lm.predict(X)
b0 = lm.intercept_
b1 = lm.coef_
estimated = b0 + b1*X

ax1 = sns.distplot(df["price"],hist = False, color="r", label="Actual Value")
sns.distplot(Yhat1, hist = False, color="b", label="Fitted Value", ax=ax1)
plt.ylim(0,)
plt.show()

'''

# Multiple Linear Regression with Distribution plot
'''

lm = LinearRegression()
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
Y=df["price"]
lm.fit(Z,Y)
Y=df["price"]
Yhat2 = lm.predict(Z)

ax1 = sns.distplot(df["price"],hist = False, color="r", label="Actual Value")
sns.distplot(Yhat2, hist = False, color="b", label="Fitted Value", ax=ax1)
plt.ylim(0,)
plt.show()

'''

# Residual Plot
'''
sns.residplot(df["highway-mpg"], df["price"])
plt.xlabel("highway-mpg")
plt.ylabel("price")
plt.show()
'''

