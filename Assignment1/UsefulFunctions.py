import pandas as pd
import os


def PricesDK(df_prices):

    df_prices["Spot"] = df_prices["SpotPriceDKK"]

    ### Calculate the fixed Tax column ###

    df_prices["Tax"] = 0.8

    ### Calculate the fixed TSO column ###

    df_prices["TSO"] = 0.1

    ### Add the DSO tariffs ###

    # Create a new column with Low load values
    df_prices["DSO"] = 0.15

    # List of winter months
    winter = [1, 2, 3, 10, 11, 12]
    peak_hours = range(17, 22)

    ## Add High load values when outside low load period
    # Winter
    df_prices.loc[
        (df_prices["HourDK"]).dt.month.isin(winter)
        & ((df_prices["HourDK"]).dt.hour >= 6),
        "DSO",
    ] = 0.45
    # Summer
    df_prices.loc[
        ((df_prices["HourDK"]).dt.month.isin(winter) == False)
        & ((df_prices["HourDK"]).dt.hour >= 6),
        "DSO",
    ] = 0.23
    ## Add Peak load values to appropriate hours
    # Winter
    df_prices.loc[
        (df_prices["HourDK"]).dt.month.isin(winter)
        & ((df_prices["HourDK"]).dt.hour.isin(peak_hours)),
        "DSO",
    ] = 1.35
    # Summer
    df_prices.loc[
        ((df_prices["HourDK"]).dt.month.isin(winter) == False)
        & ((df_prices["HourDK"]).dt.hour.isin(peak_hours)),
        "DSO",
    ] = 0.60

    ### Calculate VAT ###
    # Danish VAT is 25%
    df_prices["VAT"] = 0.25 * (
        df_prices["SpotPriceDKK"]
        + df_prices["Tax"]
        + df_prices["TSO"]
        + df_prices["DSO"]
    )

    ### Calculate Buy price ###
    # Buy price is sum of all other prices
    # This is equivalent to VAT * 5
    df_prices["Buy"] = df_prices["VAT"] * 5

    return df_prices


def LoadPriceData(filename="ElspotpricesEA.csv"):
    ### Load electricity prices ###
    price_path = os.path.join(os.getcwd(), filename)
    df_prices = pd.read_csv(price_path)

    ### Convert to datetime ###
    df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])
    df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
    df_prices["HourUTC"] = df_prices["HourUTC"].dt.tz_localize("UTC")
    df_prices["HourDK"] = df_prices["HourUTC"].dt.tz_convert("CET")

    ### Convert prices from DKK/MWh to DKK/kWh ###
    df_prices["SpotPriceDKK"] = df_prices["SpotPriceDKK"] / 1000

    ### Filter only DK2 prices ###
    df_prices = df_prices.loc[df_prices["PriceArea"] == "DK2"]

    ### Keep only the local time and price columns ###
    df_prices = df_prices[["HourDK", "SpotPriceDKK"]]

    ### Reset the index ###
    df_prices = df_prices.reset_index(drop=True)

    return df_prices


def LoadProsumerData(filename="ProsumerHourly.csv"):
    ### Load electricity prices ###
    price_path = os.path.join(os.getcwd(), filename)
    df_prices = pd.read_csv(price_path)

    ### Convert to datetime ###
    # df_prices["HourDK"] = pd.to_datetime(df_prices["HourDK"])
    # df_prices["HourUTC"] = pd.to_datetime(df_prices["HourUTC"])
    # df_prices["HourUTC"] = df_prices["HourUTC"].dt.tz_localize("UTC")
    # df_prices["HourDK"] = df_prices["HourUTC"].dt.tz_convert("CET")

    df_prices["TimeDK"] = pd.to_datetime(df_prices["TimeDK"])
    df_prices["TimeUTC"] = pd.to_datetime(df_prices["TimeUTC"])
    df_prices["TimeUTC"] = df_prices["TimeUTC"].dt.tz_localize("UTC")
    df_prices["TimeDK"] = df_prices["TimeUTC"].dt.tz_convert("CET")

    ### Convert prices from DKK/MWh to DKK/kWh ###
    ##df_prices["SpotPriceDKK"] = df_prices["SpotPriceDKK"] / 1000

    ### Filter only DK2 prices ###
    # df_prices = df_prices.loc[df_prices["PriceArea"] == "DK2"]

    ### Keep only the local time and price columns ###
    df_prices = df_prices[["TimeDK", "Consumption", "PV"]]

    ### Reset the index ###
    df_prices = df_prices.reset_index(drop=True)

    return df_prices
