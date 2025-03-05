# %% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy as cp
from UsefulFunctions import PricesDK, LoadPriceData, LoadProsumerData

# %% Load Data
df_prices = LoadPriceData()
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
C_0 = 0.5 * capacity
C_n = 0.5 * capacity
n = 24  # hours
eta_d_inv = 1 / eta_d

prices = df_prices["Spot"].values
EOD = df_prices["hour"] == 23

n = len(prices)
p_c = cp.Variable(n)
p_d = cp.Variable(n)
X = cp.Variable(n)
d = cp.Variable(n, boolean=True)

profit = cp.sum(prices @ (p_d - p_c))

constraints = [
    p_c >= 0,
    p_d >= 0,
    p_c <= d * P_max,
    p_d <= (1 - d) * P_max,
    X[0] == C_0 + p_c[0] * eta_c - p_d[0] * eta_d_inv,
    X[EOD] == C_0,
    X[-1] == C_n,
    X >= SOC_min * capacity,
    X <= SOC_max * capacity,
]
print("making steps")
for i in range(1, n):
    constraints += [
        X[i] == X[i - 1] + p_c[i] * eta_c - p_d[i] * eta_d_inv,
    ]
print("making problem")
problem = cp.Problem(cp.Maximize(profit), constraints)
print("solving")
problem.solve()
print("done")
# %%
# Create the figure and axes objects for the two subplots
pltrange = (0, 24 * 7)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig.subplots_adjust(hspace=0.4)  # Adjust space between plots

# Plot the prices in the top subplot (exclude the first and last hours)
ax1.stairs(
    prices[pltrange[0] : pltrange[1]],
    range(len(prices) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Prices",
    baseline=None,
    color="darkblue",
    linewidth=2,
)
ax1.set_xlabel("Hour", fontsize=12)
ax1.set_ylabel("Price [DKK/kWh]", fontsize=12)
ax1.set_title("Spot Prices Over Time", fontsize=14, fontweight="bold")
ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

# Plot the power in the bottom subplot (exclude the first and last hours)
ax2.stairs(
    p_c.value[pltrange[0] : pltrange[1]],
    range(len(prices) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Charging Power",
    baseline=None,
    color="green",
    linewidth=2,
)
ax2.stairs(
    -p_d.value[pltrange[0] : pltrange[1]],
    range(len(prices) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Discharging Power",
    baseline=None,
    color="red",
    linewidth=2,
)
ax2.set_xlabel("Hour", fontsize=12)
ax2.set_ylabel("Power [kW]", fontsize=12)
ax2.set_title("Battery Charging/Discharging Schedule", fontsize=14, fontweight="bold")
ax2.legend(loc="upper center", fontsize=10, frameon=True, shadow=True, ncol=2)
ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

# Plot the state of charge in the middle subplot (exclude the first and last hours)
plt.plot(
    range(len(prices + 1))[pltrange[0] : pltrange[1] + 1],
    np.insert(X.value, 0, C_0)[pltrange[0] : pltrange[1] + 1],
    label="SOC evolution",
    color="b",
    marker="o",
    linestyle="--",
)
ax3.set_xlabel("Hour", fontsize=12)
ax3.set_ylabel("State of Charge [kWh]", fontsize=12)
ax3.set_title("Battery State of Charge Over Time", fontsize=14, fontweight="bold")
ax3.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

# Show the plot
plt.tight_layout()
plt.show()
# %%
df_arbitrage = pd.DataFrame(
    {
        "Charging Power [kW]": p_c.value,
        "Discharging Power [kW]": p_d.value,
        "State of Charge [kWh]": X.value,
    }
)
df_arbitrage["Profit [DKK]"] = (
    -df_arbitrage["Charging Power [kW]"] * prices
    + df_arbitrage["Discharging Power [kW]"] * prices
)
df_arbitrage["Price [DKK/kWh]"] = prices
df_arbitrage["Time"] = df_prices["HourDK"]
df_arbitrage.where(
    df_arbitrage.loc[df_arbitrage["Time"].dt.hour == 23, "State of Charge [kWh]"] != C_0
).dropna()
df_arbitrage["charge total [kW]"] = (
    df_arbitrage["Charging Power [kW]"] - df_arbitrage["Discharging Power [kW]"]
)
df_arbitrage_year = (
    df_arbitrage[["Profit [DKK]", "Charging Power [kW]", "Discharging Power [kW]"]]
    .groupby(df_arbitrage["Time"].dt.year)
    .sum()
)
df_arbitrage_day = (
    df_arbitrage[["Profit [DKK]", "Charging Power [kW]", "Discharging Power [kW]"]]
    .groupby(df_arbitrage["Time"].dt.date)
    .sum()
)
df_arbitrage_day["charge total [kW]"] = (
    df_arbitrage[["charge total [kW]"]].groupby(df_arbitrage["Time"].dt.date).sum()
)
df_arbitrage_day["price SD"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).std()
)
df_arbitrage_day["price var"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).var()
)
df_arbitrage_day["price mean"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).mean()
)
df_arbitrage_day["price max"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).max()
)
df_arbitrage_day["price min"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).min()
)
df_arbitrage_day["price span"] = (
    df_arbitrage_day["price max"] - df_arbitrage_day["price min"]
)

for col in df_arbitrage_day.loc[:, df_arbitrage_day.columns.str.startswith("price")]:
    df_arbitrage_day.plot(
        x=col, y="Profit [DKK]", kind="scatter", title=f"Profit vs {col}", grid=True
    )


# %% Task 3
df_prices = LoadPriceData()
df_prices = PricesDK(df_prices)
PH_prices = LoadProsumerData()
df_prices = df_prices[["HourDK", "Spot", "Buy"]]
PH_prices = PH_prices[["Consumption"]]
df_prices = df_prices[
    (df_prices["HourDK"].dt.year >= 2022) & (df_prices["HourDK"].dt.year <= 2023)
].reset_index()
df_prices["Spot"] = df_prices["Spot"]
df_prices.rename(columns={"Spot": "Sell"}, inplace=True)
df_prices["Consumption"] = PH_prices[["Consumption"]]

# Explicitly extract the year and hour from the datetime column
df_prices["year"] = df_prices["HourDK"].dt.year
# df_prices["hour"] = df_prices["HourDK"].dt.hour

df_mean = (
    df_prices.groupby(["year"])
    .agg({"Buy": "mean", "Sell": "mean", "Consumption": "sum"})
    .reset_index()
)
df_mean["Cost"] = df_mean["Buy"] * df_mean["Consumption"]

df_prices["hourly cost"] = df_prices["Buy"] * df_prices["Consumption"]
df_mean_1 = df_prices.groupby(["year"]).agg({"hourly cost": "sum"}).reset_index()
df_mean["Hourly Cost"] = df_mean_1["hourly cost"]

# df_prices = df_prices[["TimeDK", "Consumption"]]

# main - 2 dataframes, priser, consumption(pv),
# i df pricces, ny kolonne - consumtion fra den anden dataframe.

# result = pd.merge(df_prices, on='HourDK', how='inner')
