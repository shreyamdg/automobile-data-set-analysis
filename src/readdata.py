import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

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

##Exporting to csv
"""

path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)

"""

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
#print(df.dtypes)



###Data Binning

binwidth = int((max(df["price"])-min(df["price"]))/3)
bins = range(min(df["price"]), max(df["price"]), binwidth)
group_names = ['low','medium','high']
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)
df.dropna(subset=["price-binned"], axis=0, inplace=True)

##Plotting Histogram from the binned value

plt.hist(df["price"],bins=3)
plt.title("Price Bins")
plt.xlabel("Count")
plt.ylabel("Price")
plt.show()