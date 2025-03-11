# %% Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cvxpy as cp
import scipy.optimize as opt
from UsefulFunctions import PricesDK, LoadPriceData, LoadProsumerData, Netting

# %% Load Data
df_prices = LoadPriceData()
df_prices = PricesDK(df_prices)
df_prices.head()
df_pro = LoadProsumerData()
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
# %% Task 2.1 Optimization
### Define the given parameters ###
## SOC limits
SOC_min = 0.1
SOC_max = 1
## Power and energy limits
P_max = 5  # kW
capacity = 10  # kWh
## Efficiencies
eta_c = eta_d = 0.95
# Inverse of the discharge efficiency to avoid division
eta_d_inv = 1 / eta_d
## Initial and final SOC
C_0 = 0.5 * capacity
C_n = 0.5 * capacity
## Data series
### Data series
prices = df_prices["Spot"].values
EOD = df_prices["hour"] == 23

### Create the optimization variables
## Number of steps in the optimization = number of hours in data
n = len(prices)
## Charging and discharging power per hour
p_c = cp.Variable(n)
p_d = cp.Variable(n)
## Battery state of charge per hour
X = cp.Variable(n)
## Boolean variable to indicate if the battery is charging
d = cp.Variable(n, boolean=True)

### Define the variable to be optimized
profit = cp.sum(prices @ (p_d - p_c))

### Define the constraints
constraints = [
    ## Power constraints
    p_c >= 0,
    p_d >= 0,
    p_c <= d * P_max,
    p_d <= (1 - d) * P_max,
    ## State of charge constraints
    # Charging in initial time step
    X[0] == C_0 + p_c[0] * eta_c - p_d[0] * eta_d_inv,
    # Constrain SOC at End of Day to be C_n every day
    X[EOD] == C_n,
    # Final SOC constraint
    X[-1] == C_n,
    # SOC limits
    X >= SOC_min * capacity,
    X <= SOC_max * capacity,
]
### Make SOC time steps
print("making steps")
for i in range(1, n):
    constraints += [
        X[i] == X[i - 1] + p_c[i] * eta_c - p_d[i] * eta_d_inv,
    ]
### Create the optimization problem
print("making problem")
problem = cp.Problem(cp.Maximize(profit), constraints)
### Solve the optimization problem
print("solving")
problem.solve()
print("done")
# %% Task 2.1 debugging plot
### Debugging plot to check the optimization results ###
# Limit range of plot to show only a few days
pltrange = (0, 24 * 7)
# Create the figure and axes objects for the two subplots
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

# Plot the power in the middle subplot (exclude the first and last hours)
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

# Plot the state of charge in the bottom subplot
ax3.plot(
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

ax3.set_xlim(pltrange)
ax3.set_xticks(range(pltrange[0], pltrange[1] + 1, 12))
# Show the plot
plt.tight_layout()
plt.show()
# %% Task 2.1 data analysis
### Create the dataframe with the optimization results ###
df_arbitrage = pd.DataFrame(
    {
        "Charging Power [kW]": p_c.value,
        "Discharging Power [kW]": p_d.value,
        "State of Charge [kWh]": X.value,
    }
)
## Calculate profits of each hour
df_arbitrage["Profit [DKK]"] = (
    -df_arbitrage["Charging Power [kW]"] * prices
    + df_arbitrage["Discharging Power [kW]"] * prices
)
df_arbitrage["Price [DKK/kWh]"] = prices
df_arbitrage["Time"] = df_prices["HourDK"]
### Check constraints for the state of charge at EOD fits###
## This should make an empty data frame
## All non-empty entries are violations of the constraint
df_arbitrage.where(
    df_arbitrage.loc[df_arbitrage["Time"].dt.hour == 23, "State of Charge [kWh]"] != C_0
).dropna()
df_arbitrage["charge total [kW]"] = (
    df_arbitrage["Charging Power [kW]"] - df_arbitrage["Discharging Power [kW]"]
)
## Yearly aggregation
df_arbitrage_year = (
    df_arbitrage[["Profit [DKK]", "Charging Power [kW]", "Discharging Power [kW]"]]
    .groupby(df_arbitrage["Time"].dt.year)
    .sum()
)
plt.figure()
plt.bar(df_arbitrage_year.index, df_arbitrage_year["Profit [DKK]"])
plt.title("Yearly profit from arbitrage")
plt.xlabel("Year")
plt.ylabel("Profit [DKK]")
plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
plt.yticks(np.arange(0, 4501, 500))
plt.tight_layout()
plt.show()

# %% Task 2.2
## Daily aggregation
df_arbitrage_day = (
    df_arbitrage[["Profit [DKK]", "Charging Power [kW]", "Discharging Power [kW]"]]
    .groupby(df_arbitrage["Time"].dt.date)
    .sum()
)
### Make statistics metrics for the daily data ###
## Total charge
df_arbitrage_day["charge total [kW]"] = (
    df_arbitrage[["charge total [kW]"]].groupby(df_arbitrage["Time"].dt.date).sum()
)
## Standard deviation of the price
df_arbitrage_day["price SD [DKK/kWh]"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).std()
)
## Variance of the price
df_arbitrage_day["price var [(DKK/kWh)^2]"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).var()
)
## Mean price
df_arbitrage_day["price mean [DKK/kWh]"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).mean()
)
## Max price
df_arbitrage_day["price max [DKK/kWh]"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).max()
)
## Min price
df_arbitrage_day["price min [DKK/kWh]"] = (
    df_arbitrage[["Price [DKK/kWh]"]].groupby(df_arbitrage["Time"].dt.date).min()
)
## Price span
df_arbitrage_day["price span [DKK/kWh]"] = (
    df_arbitrage_day["price max [DKK/kWh]"] - df_arbitrage_day["price min [DKK/kWh]"]
)
""" ## Coefficient of variance
df_arbitrage_day["price cov"] = (
    df_arbitrage_day["price SD"] / df_arbitrage_day["price mean"]
) """
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Daily statistics for arbitrage", fontsize=20)
## Plotting the daily statistics
for i, col in enumerate(
    df_arbitrage_day.loc[:, df_arbitrage_day.columns.str.startswith("price")]
):
    # Do linear regression
    m, b = np.polyfit(df_arbitrage_day[col], df_arbitrage_day["Profit [DKK]"], 1)
    # Plot the scatter plot
    axs[i // 3, i % 3].scatter(
        df_arbitrage_day[col], df_arbitrage_day["Profit [DKK]"], label="Data"
    )
    # Plot the linear regression
    axs[i // 3, i % 3].plot(
        df_arbitrage_day[col],
        m * df_arbitrage_day[col] + b,
        color="red",
        label="Linear regression",
    )
    # Calculate r^2
    residuals = df_arbitrage_day["Profit [DKK]"] - (m * df_arbitrage_day[col] + b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (df_arbitrage_day["Profit [DKK]"] - np.mean(df_arbitrage_day["Profit [DKK]"]))
        ** 2
    )
    r_squared = 1 - (ss_res / ss_tot)
    # Set the title
    axs[i // 3, i % 3].set_title(
        "{} ($R^2={}$)".format(col, round(r_squared, 2)), fontsize=16
    )
    # Set the labels
    axs[i // 3, i % 3].set_xlabel(col, fontsize=16)
    axs[i // 3, i % 3].set_ylabel("Profit [DKK]", fontsize=16)
    # Set the legend
    # axs[i//3, i%3].legend(fontsize=14)
fig.legend(
    ["Data", "Linear regression"],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.005),
    ncol=2,
    fontsize=14,
)
plt.tight_layout()
plt.show()

# %% Task 2.3
df_sot = pd.DataFrame(index=range(2019, 2024))
for j in range(1, 11):
    ### Define the given parameters ###
    ## SOC limits
    SOC_min = 0.1
    SOC_max = 1
    ## Power and energy limits
    P_max = 5  # kW
    capacity = 10  # kWh
    ## Efficiencies
    eta_c = eta_d = 0.95
    # Inverse of the discharge efficiency to avoid division
    eta_d_inv = 1 / eta_d
    ## Initial and final SOC
    C_0 = j / 10 * capacity
    C_n = j / 10 * capacity
    ## Data series
    prices = df_prices["Spot"].values
    EOD = df_prices["hour"] == 23

    ### Create the optimization variables
    ## Number of steps in the optimization = number of hours in data
    n = len(prices)
    ## Charging and discharging power per hour
    p_c = cp.Variable(n)
    p_d = cp.Variable(n)
    ## Battery state of charge per hour
    X = cp.Variable(n)
    ## Boolean variable to indicate if the battery is charging
    d = cp.Variable(n, boolean=True)

    ### Define the variable to be optimized
    profit = cp.sum(prices @ (p_d - p_c))

    ### Define the constraints
    constraints = [
        ## Power constraints
        p_c >= 0,
        p_d >= 0,
        p_c <= d * P_max,
        p_d <= (1 - d) * P_max,
        ## State of charge constraints
        # Charging in initial time step
        X[0] == C_0 + p_c[0] * eta_c - p_d[0] * eta_d_inv,
        # Constrain SOC at End of Day to be C_n every day
        X[EOD] == C_n,
        # Final SOC constraint
        X[-1] == C_n,
        # SOC limits
        X >= SOC_min * capacity,
        X <= SOC_max * capacity,
    ]
    ### Make SOC time steps
    print("making steps")
    for i in range(1, n):
        constraints += [
            X[i] == X[i - 1] + p_c[i] * eta_c - p_d[i] * eta_d_inv,
        ]
    ### Create the optimization problem
    print("making problem")
    problem = cp.Problem(cp.Maximize(profit), constraints)
    ### Solve the optimization problem
    print("solving")
    problem.solve()
    print("done")
    df_temp = pd.DataFrame(
        {
            "Charging Power [kW]": p_c.value,
            "Discharging Power [kW]": p_d.value,
            "State of Charge [kWh]": X.value,
        }
    )
    ## Calculate profits of each hour
    df_temp["Profit [DKK]"] = (
        -df_temp["Charging Power [kW]"] * prices
        + df_temp["Discharging Power [kW]"] * prices
    )
    df_temp["Price [DKK/kWh]"] = prices
    df_temp["Time"] = df_prices["HourDK"]
    df_sot["{}%".format(j * 10)] = (
        df_temp["Profit [DKK]"].groupby(df_temp["Time"].dt.year).sum().values
    )
    print(df_sot)

# %% Task 2.3 plot
plt.figure()
# Grouped bar chart with the profit for each charging level
x = len(df_sot.columns)
bars = len(df_sot.index)
width = 1 / (bars + 1)
multiplier = 0
for year in df_sot.index:
    plt.bar(
        np.arange(x) + width * (multiplier - (bars - 1) / 2),
        df_sot.loc[year].values,
        width=width,
        label=year,
    )
    multiplier += 1
plt.xticks(np.arange(x), df_sot.columns)
plt.xlabel("$C_n$ as a percentage of capacity")
plt.ylabel("Profit [DKK]")
plt.title("Yearly profit from arbitrage at different $C_n$ constraints")
plt.legend(title="Year", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
plt.tight_layout()
plt.show()

# %% Task 2.3 dynamic SOT
### Define the given parameters ###
## SOC limits
SOC_min = 0.1
SOC_max = 1
## Power and energy limits
P_max = 5  # kW
capacity = 10  # kWh
## Efficiencies
eta_c = eta_d = 0.95
# Inverse of the discharge efficiency to avoid division
eta_d_inv = 1 / eta_d
## Initial and final SOC
C_0 = 0.5 * capacity
## Data series
prices = df_prices["Spot"].values
EOD = df_prices["hour"] == 23

### Create the optimization variables
## Number of steps in the optimization = number of hours in data
n = len(prices)
## Charging and discharging power per hour
p_c = cp.Variable(n)
p_d = cp.Variable(n)
## Battery state of charge per hour
X = cp.Variable(n)
## Boolean variable to indicate if the battery is charging
d = cp.Variable(n, boolean=True)

### Define the variable to be optimized
profit = cp.sum(prices @ (p_d - p_c))

### Define the constraints
constraints = [
    ## Power constraints
    p_c >= 0,
    p_d >= 0,
    p_c <= d * P_max,
    p_d <= (1 - d) * P_max,
    ## State of charge constraints
    # Charging in initial time step
    X[0] == C_0 + p_c[0] * eta_c - p_d[0] * eta_d_inv,
    # SOC limits
    X >= SOC_min * capacity,
    X <= SOC_max * capacity,
]
### Make SOC time steps
print("making steps")
for i in range(1, n):
    constraints += [
        X[i] == X[i - 1] + p_c[i] * eta_c - p_d[i] * eta_d_inv,
    ]
### Create the optimization problem
print("making problem")
problem = cp.Problem(cp.Maximize(profit), constraints)
### Solve the optimization problem
print("solving")
problem.solve()
print("done")

# %% Task 2.3 dynamic SOT data analysis
### Create the dataframe with the optimization results ###
df_dynamic = pd.DataFrame(
    {
        "Charging Power [kW]": p_c.value,
        "Discharging Power [kW]": p_d.value,
        "State of Charge [kWh]": X.value,
    }
)
## Calculate profits of each hour
df_dynamic["Profit [DKK]"] = (
    -df_dynamic["Charging Power [kW]"] * prices
    + df_dynamic["Discharging Power [kW]"] * prices
)
df_dynamic["Price [DKK/kWh]"] = prices
df_dynamic["Time"] = df_prices["HourDK"]
df_sot["dynamic"] = (
    df_dynamic["Profit [DKK]"].groupby(df_dynamic["Time"].dt.year).sum().values
)
X[EOD].value

plt.figure(figsize=(8, 5))
# Grouped bar chart with the profit for each charging level
x = len(df_sot.columns)
bars = len(df_sot.index)
width = 1 / (bars + 1)
multiplier = 0
for year in df_sot.index:
    plt.bar(
        np.arange(x) + width * (multiplier - (bars - 1) / 2),
        df_sot.loc[year].values,
        width=width,
        label=year,
    )
    multiplier += 1
plt.xticks(np.arange(x), df_sot.columns)
plt.xlabel("$C_n$ as a percentage of capacity", fontdict={"fontsize": 12})
plt.ylabel("Profit [DKK]", fontdict={"fontsize": 12})
plt.title(
    "Yearly profit from arbitrage at different $C_n$ constraints",
    fontdict={"fontsize": 14},
)
plt.legend(title="Year", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
plt.tight_layout()
plt.show()

plt.figure()
for year in df_sot.index:
    df_temp = df_dynamic.loc[
        (df_dynamic["Time"].dt.year == year) & (df_dynamic["Time"].dt.hour == 23)
    ]
    plt.boxplot(
        df_temp["State of Charge [kWh]"] / capacity * 100,
        positions=[year],
        widths=0.6,
        patch_artist=True,
    )
plt.xlabel("Year", fontdict={"fontsize": 12})
plt.ylabel("State of Charge [%]", fontdict={"fontsize": 12})
plt.yticks(np.arange(0, 101, 10))
plt.title("Dynamic State of Charge at the end of the day", fontdict={"fontsize": 14})
plt.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.8)
plt.tight_layout()
plt.show()

# %% Task 3.1
### Load Data ###
## Load the price data
df_prices = LoadPriceData()
df_prices = PricesDK(df_prices)
## Load the prosumer data
PH_prices = LoadProsumerData()
## Limit the data to the relevant columns and years
# Relevant columns
df_prices = df_prices[["HourDK", "Spot", "Buy"]]
PH_prices = PH_prices[["Consumption", "PV"]]
# Limit years
df_prices = df_prices[
    (df_prices["HourDK"].dt.year >= 2022) & (df_prices["HourDK"].dt.year <= 2023)
].reset_index()
df_prices.rename(columns={"Spot": "Sell"}, inplace=True)
## Add prosumer data to the price data
df_prices["Consumption"] = PH_prices[["Consumption"]]
df_prices["PV"] = PH_prices[["PV"]]
df_prices["Net"] = df_prices["Consumption"] - df_prices["PV"]

## Explicitly extract the year from the datetime column
df_prices["year"] = df_prices["HourDK"].dt.year

### Make aggregate dataframes for the mean prices and total consumption ###
df_aggregate = (
    df_prices.groupby(["year"])
    .agg({"Buy": "mean", "Sell": "mean", "Consumption": "sum"})
    .reset_index()
)
## Total cost for each year
## as the product of the average buy price and total consumption
df_aggregate["Cost"] = df_aggregate["Buy"] * df_aggregate["Consumption"]

## Calculate the hourly cost for each hour
df_prices["hourly cost"] = df_prices["Buy"] * df_prices["Consumption"]
# Aggregate the hourly cost to yearly level
df_aggregate["Hourly Cost"] = (
    df_prices.groupby(["year"]).agg({"hourly cost": "sum"}).values
)

# df_prices = df_prices[["TimeDK", "Consumption"]]

# main - 2 dataframes, priser, consumption(pv),
# i df pricces, ny kolonne - consumtion fra den anden dataframe.


# result = pd.merge(df_prices, on='HourDK', how='inner')
# %% Task 3.2
df_pro = df_pro[["HourDK", "PV", "Consumption"]]
df_pro["Month"] = df_pro["HourDK"].dt.month
df_pro["Year"] = df_pro["HourDK"].dt.year
df_pro["DayOfMonth"] = df_pro["HourDK"].dt.day
df_pro["DayOfYear"] = df_pro["HourDK"].dt.dayofyear


df_prices = df_prices[["HourDK", "Sell", "Buy", "Net"]]
df_pro = LoadProsumerData()
df_pro.rename(columns={"TimeDK": "HourDK"}, inplace=True)
### Merge the dataframes to create 'result' ###
result = pd.merge(df_prices, df_pro, on="HourDK", how="inner")

### Add time features ###
result["Month"] = result["HourDK"].dt.month
result["Year"] = result["HourDK"].dt.year
result["DayOfMonth"] = result["HourDK"].dt.day
result["DayOfYear"] = result["HourDK"].dt.dayofyear


res = "Hourly"
Net_Hourly = Netting(result, res)
print("\nThe hourly netting results are:\n", Net_Hourly)

res = "Yearly"
Net_Yearly = Netting(result, res)
print("\nThe yearly netting results are:\n", Net_Yearly)

# benefit on a yearly basis
### Extract Yearly Costs Without PV
df_hourly_cost = df_aggregate[["year", "Hourly Cost"]].rename(columns={"year": "Year"})

# Merge Hourly Cost with Netting Profit
df_comparison = pd.merge(Net_Hourly, df_hourly_cost, on="Year", how="inner")

# Correct Net Benefit Calculation
df_comparison["Net Benefit"] = (
    df_comparison["Profit"] + df_comparison["Hourly Cost"]
)  # Since Cost is an expense

# Display the comparison
print("\nComparison of Hourly Netting Profit and Hourly Cost:\n", df_comparison)

# %% Improved Bar Plot
plt.figure(figsize=(8, 5))

# Define x positions for bars
x_years = df_comparison["Year"]

# Plot bars for Netting Profit and Hourly Cost
plt.bar(
    x_years - 0.2,
    df_comparison["Profit"],
    width=0.4,
    label="Netting Profit",
    color="blue",
)
plt.bar(
    x_years + 0.2,
    -df_comparison["Hourly Cost"],
    width=0.4,
    label="Hourly Cost (as expense)",
    color="red",
)  # Negative to reflect cost

# Formatting
plt.xlabel("Year")
plt.ylabel("Amount (DKK)")
plt.title("Comparison of Hourly Netting Profit vs. Hourly Cost")
plt.axhline(0, color="black", linewidth=1)  # Zero-line for reference
plt.xticks([2022, 2023], labels=["2022", "2023"])  # Ensure only 2022 and 2023 appear
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.show()


# %% over 20 år

pv_system_cost = 85000  # 85,000 DKK (example assumption)

# Compute Total Savings Over 20
df_comparison["Total_Savings_20Y"] = (df_comparison["Net Benefit"] * 20).round(2)

# Compute Payback Period
df_comparison["Payback_Period"] = (pv_system_cost / df_comparison["Net Benefit"]).round(
    2
).astype(str) + " years"

print("\nPV System Investment Analysis Over 20 Years:")
print(df_comparison[["Year", "Net Benefit", "Total_Savings_20Y", "Payback_Period"]])

avg_net_benefit = df_comparison["Net Benefit"].mean()
avg_total_savings_20Y = avg_net_benefit * 20
avg_payback_period = pv_system_cost / avg_net_benefit


print(f"Average Net Benefit: {avg_net_benefit:.2f} DKK")
print(f"Average Total Savings Over 20 Years: {avg_total_savings_20Y:.2f} DKK")
print(f"Average Payback Period: {avg_payback_period:.2f} years")


# %% Task 3.3
### Define the given parameters ###
## SOC limits
SOC_min = 0.1
SOC_max = 1
## Power and energy limits
P_max = 5  # kW
capacity = 10  # kWh
## Efficiencies
eta_c = eta_d = 0.95
# Inverse of the discharge efficiency to avoid division
eta_d_inv = 1 / eta_d
## Initial and final SOC
C_0 = 0.5 * capacity
C_n = 0.5 * capacity
### Data series
## Prices
sell = df_prices["Sell"].values
buy = df_prices["Buy"].values
## Net consumption
net = df_prices["Net"].values
## EOD indicator for the last hour of the day
EOD = df_prices["HourDK"].dt.hour == 23
## Energy Surplus and shortage
surplus = (-df_prices["Net"]).clip(lower=0).values
shortage = (df_prices["Net"]).clip(lower=0).values

### Create the optimization variables
## Number of steps in the optimization = number of hours in data
n = len(net)
## Charging and discharging power per hour
p_c = cp.Variable(n)
p_d = cp.Variable(n)
## Import and export power per hour
imp = cp.Variable(n)
exp = cp.Variable(n)
## Battery state of charge per hour
X = cp.Variable(n)

### Define the variable to be optimized
cost = cp.sum(-exp @ sell + imp @ buy)

### Define the constraints
constraints = [
    ## Power constraints
    p_c >= 0,
    p_d >= 0,
    imp >= 0,
    exp >= 0,
    p_c <= P_max,
    p_d <= P_max,
    ## Relationship between import/export and charging/discharging
    ## Sum of battery power and shortage/surplus must equal import/export
    surplus + p_d + imp == p_c + shortage + exp,
    ## State of charge constraints
    # Charging in initial time step
    X[0] == C_0 + p_c[0] * eta_c - p_d[0] * eta_d_inv,
    # Constrain SOC at End of Day to be C_n every day
    X[EOD] == C_n,
    # Final SOC constraint
    X[-1] == C_n,
    # SOC limits
    X >= SOC_min * capacity,
    X <= SOC_max * capacity,
]
### Make SOC time steps
print("making steps")
for i in range(1, n):
    constraints += [X[i] == X[i - 1] + p_c[i] * eta_c - p_d[i] * eta_d_inv]
### Create the optimization problem as a minimization problem
print("making problem")
problem = cp.Problem(cp.Minimize(cost), constraints)
### Solve the optimization problem
print("solving")
problem.solve()
print("done")
print(problem.status)

# %% Task 3.3 debugging plot

# %% Task 3.3 data analysis
### Create the dataframe with the optimization results ###
df_prosumer = pd.DataFrame(
    {
        "Charging Power [kW]": p_c.value,
        "Discharging Power [kW]": p_d.value,
        "State of Charge [kWh]": X.value,
        "Import Power [kW]": imp.value,
        "Export Power [kW]": exp.value,
    }
)
## Calculate profits of each hour
df_prosumer["Cost [DKK]"] = (
    -df_prosumer["Export Power [kW]"] * sell + df_prosumer["Import Power [kW]"] * buy
)
df_prosumer["Time"] = df_prices["HourDK"]
print(df_prosumer["Cost [DKK]"].groupby(df_prosumer["Time"].dt.year).sum())

# %% Task 3.3 debugging plot
### Debugging plot to check the optimization results ###
# Create the figure and axes objects for the two subplots
pltrange = (4008, 4008 + 24 * 2)
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
fig.subplots_adjust(hspace=0.4)  # Adjust space between plots

# Plot the prices in the top subplot (exclude the first and last hours)
ax1.stairs(
    buy[pltrange[0] : pltrange[1]],
    range(len(buy) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Prices",
    baseline=None,
    color="darkgreen",
    linewidth=2,
)
ax1.stairs(
    sell[pltrange[0] : pltrange[1]],
    range(len(sell) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Prices",
    baseline=None,
    color="darkblue",
    linewidth=2,
)
ax1.set_xlabel("Hour", fontsize=12)
ax1.set_ylabel("Price [DKK/kWh]", fontsize=12)
ax1.set_title("Spot Prices Over Time", fontsize=14, fontweight="bold")
ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

# Plot the power in the 2nd subplot
ax2.stairs(
    p_c.value[pltrange[0] : pltrange[1]],
    range(len(p_d.value) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Charging Power",
    baseline=None,
    color="green",
    linewidth=2,
)
ax2.stairs(
    -p_d.value[pltrange[0] : pltrange[1]],
    range(len(p_d.value) + 1)[pltrange[0] : pltrange[1] + 1],
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

# Plot the state of charge in the 3rd subplot
ax3.plot(
    range(len(p_d.value))[pltrange[0] : pltrange[1] + 1],
    np.insert(X.value, 0, C_0)[pltrange[0] : pltrange[1] + 1] / capacity,
    label="SOC evolution",
    color="b",
    marker="o",
    linestyle="--",
)
ax3.set_xlabel("Hour", fontsize=12)
ax3.set_ylabel("State of Charge [kWh]", fontsize=12)
ax3.set_title("Battery State of Charge Over Time", fontsize=14, fontweight="bold")
ax3.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

# Plot the import/export in the 4th subplot
ax4.stairs(
    imp.value[pltrange[0] : pltrange[1]],
    range(len(p_d.value) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Import",
    baseline=None,
    color="green",
    linewidth=2,
)
ax4.stairs(
    -exp.value[pltrange[0] : pltrange[1]],
    range(len(p_d.value) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Export",
    baseline=None,
    color="red",
    linewidth=2,
)
ax4.set_xlabel("Hour", fontsize=12)
ax4.set_ylabel("Power [kW]", fontsize=12)
ax4.set_title("Import/Export Schedule", fontsize=14, fontweight="bold")
ax4.legend(loc="upper center", fontsize=10, frameon=True, shadow=True, ncol=2)
ax4.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)

# Plot the surplus/shortage in the 5th subplot
ax5.stairs(
    surplus[pltrange[0] : pltrange[1]],
    range(len(p_d.value) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Surplus",
    baseline=None,
    color="green",
    linewidth=2,
)
ax5.stairs(
    -shortage[pltrange[0] : pltrange[1]],
    range(len(p_d.value) + 1)[pltrange[0] : pltrange[1] + 1],
    label="Shortage",
    baseline=None,
    color="red",
    linewidth=2,
)
ax5.set_xlabel("Hour", fontsize=12)
ax5.set_ylabel("Power [kW]", fontsize=12)
ax5.set_title("Surplus/Shortage Schedule", fontsize=14, fontweight="bold")
ax5.legend(loc="upper center", fontsize=10, frameon=True, shadow=True, ncol=2)
ax5.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
ax5.set_xlim(pltrange)
ax5.set_xticks(range(pltrange[0], pltrange[1] + 1, 4))
# Show the plot
plt.tight_layout()
plt.show()

# %% Task 3.4
### Define the given parameters ###
SOC_min = 0.1
SOC_max = 1
P_max = 5  # kW
capacity = 10  # kWh
eta_c = eta_d = 0.95
C_0 = 0.5 * capacity
C_n = 0.5 * capacity
eta_d_inv = 1 / eta_d

sell = df_prices["Sell"].values
buy = df_prices["Buy"].values
net = df_prices["Consumption"].values

surplus = (-net).clip(min=0)
shortage = (net).clip(min=0)

EOD = df_prices["HourDK"].dt.hour == 23

n = len(net)
p_c = cp.Variable(n)
p_d = cp.Variable(n)
imp = cp.Variable(n)
exp = cp.Variable(n)
X = cp.Variable(n)

cost = cp.sum(-exp @ sell + imp @ buy)

constraints = [
    p_c >= 0,
    p_d >= 0,
    imp >= 0,
    exp >= 0,
    p_c <= P_max,
    p_d <= P_max,
    surplus + p_d + imp == shortage + p_c + exp,
    X[0] == C_0 + p_c[0] * eta_c - p_d[0] * eta_d_inv,
    X[EOD] == C_n,
    X[-1] == C_n,
    X >= SOC_min * capacity,
    X <= SOC_max * capacity,
]
print("making steps")
for i in range(1, n):
    constraints += [X[i] == X[i - 1] + p_c[i] * eta_c - p_d[i] * eta_d_inv]
print("making problem")
problem = cp.Problem(cp.Minimize(cost), constraints)
print("solving")
problem.solve()
print("done")
print(problem.status)
# %%

df_arb_battery = pd.DataFrame({"Import power": imp.value, "Export power": exp.value})

df_arb_battery["Cost"] = (
    df_arb_battery["Import power"] * buy - df_arb_battery["Export power"] * buy
)
df_arb_battery["Time"] = df_prices["HourDK"]

print(df_arb_battery["Cost"].groupby(df_arb_battery["Time"].dt.year).sum())
