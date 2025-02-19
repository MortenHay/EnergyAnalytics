# %% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy as cp
from UsefulFunctions import PricesDK, LoadData
#This is an update

# %% Load Data
df_prices = LoadData()
df_prices = PricesDK(df_prices)
df_prices.head()
# %% Task 1.1:
### Plotting the average spot price for each year ###
# Calculate the average spot price for each year
annual_means = df_prices["Spot"].groupby(df_prices["HourDK"].dt.year).mean()
# Make simple bar chart
plt.figure()
plt.bar(annual_means.index, annual_means * 1000)  # Convert to DKK/MWh
# Make it pretty
plt.title("Average energy price for each year")
plt.xlabel("Year")
plt.ylabel("Average price [DKK/MWh]")
plt.tight_layout()
plt.show()

# %% Task 1.2:
### Plotting the average hourly spot price for each year ###
# Explicitly extract the year and hour from the datetime column
df_prices["year"] = df_prices["HourDK"].dt.year
df_prices["hour"] = df_prices["HourDK"].dt.hour
## Make new dataframe with the average price for each year and hour
# Columns are years, rows are hours
hourly_means = pd.DataFrame(
    columns=df_prices["year"].unique(), index=df_prices["hour"].unique()
)
# Calculate the hourly means for each year
for year in hourly_means.columns:
    hourly_means[year] = (
        df_prices.loc[df_prices["year"] == year].groupby("hour")["Spot"].mean()
    )
# Copy the last row to make the final step look nice
## Make plot
plt.figure()
# Plot each year separately as a step plot
for c in hourly_means.columns:
    plt.stairs(
        hourly_means[c] * 1000,
        range(hourly_means.index[0], hourly_means.index[-1] + 2),
        label=c,
    )
# Make it pretty
plt.legend()
plt.title("Average electricity spot price for each hour of the day")
plt.xlabel("Hour of the day")
plt.ylabel("Average price [DKK/MWh]")
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 2))
plt.tight_layout()
plt.show()
# %% Task 2.1
### Define the given parameters ###
SOC_min = 0.1
SOC_max = 1
P_max = 5  # kW
capacity = 10  # kWh
eta_c = eta_d = 0.95
X_0 = 0.5
X_n = 0.5
