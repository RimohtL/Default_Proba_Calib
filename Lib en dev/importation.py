import pandas as pd
import numpy as np



def get_data(file="/Users/user/Documents/Cours/CentraleSupelec/Projet/Docs/Data issuers.xlsx",
             sheet_name_market_cap="Mod Market Cap",
             date=['2019-10-28', '2020-10-13'],
             sheet_name_debt="Gross Debt"):
    
    """
    Get market capitalization and debt data from Excel file.

    Args:
        file (str, optional): Path to the Excel file (default is "/Users/user/Documents/Cours/CentraleSupelec/Projet/Docs/Data issuers.xlsx").
        sheet_name_market_cap (str, optional): Name of the sheet containing market capitalization data (default is "Mod Market Cap"). Structure is : on the row 0 : 'Dates', ticker_names1, ticker8names2 ... / on the table: dd/mm/yyyy, float, float....
        date (list, optional): List containing two dates specifying the range of market capitalization data to retrieve (default is ['2019-10-28', '2020-10-13']).
        sheet_name_debt (str, optional): Name of the sheet containing debt data (default is "Gross Debt"). Structure is 1 row with ticker name, 1 row with debt.

    Returns:
        tuple: A tuple containing market_cap and debt DataFrames.
    """
    market_cap = pd.read_excel(file, sheet_name=sheet_name_market_cap)
    date1 = date[0]
    date2 = date[1]
    market_cap = market_cap.set_index("Dates").loc[str(date1):str(date2)]
    debt = pd.read_excel(file, sheet_name=sheet_name_debt, nrows=1)

    return market_cap, debt
