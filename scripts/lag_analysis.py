import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import grangercausalitytests

# ── File: scripts/lag_analysis.py ─────────────────────────────────────────────
# Loads two-column CSVs, runs lagged regression, cross-correlation & Granger tests

# Paths for uniform two-column CSVs
datasets = 'datasets/'
paths = {
    'box_office': f'{datasets}Daily_BoxOffice.csv',
    'netflix':    f'{datasets}Daily_Netflix.csv',
    'spy':        f'{datasets}Daily_SPX.csv',
    'gdp':        f'{datasets}Quarterly_GDP.csv'
}
# Expected date format in all CSVs
DATE_FORMAT = '%m/%d/%y'


def load_series(name):
    """
    Reads the first two columns (date and value) from CSV at paths[name],
    explicitly parses dates using DATE_FORMAT, cleans numeric values,
    and returns a pandas Series indexed by date.
    """
    # Read first two columns as strings
    df = pd.read_csv(
        paths[name],
        usecols=[0, 1],
        header=0,
        dtype={0: str, 1: str}
    )
    # Rename columns to standard names
    df.columns = ['date', name]
    # Parse date explicitly
    df['date'] = pd.to_datetime(
        df['date'],
        format=DATE_FORMAT,
        errors='coerce'
    )
    # Clean numeric column by stripping $/commas
    df[name] = pd.to_numeric(
        df[name].str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    # Drop any rows with invalid date or value
    df.dropna(subset=['date', name], inplace=True)
    # Return sorted series
    return df.set_index('date')[name].sort_index()

# Load each series
daily_box = load_series('box_office')
daily_net = load_series('netflix')
daily_spy = load_series('spy')
# Quarterly GDP (less frequent)
quarterly_gdp = load_series('gdp')

# Merge daily series on intersection of dates
daily = pd.concat(
    [daily_box, daily_net, daily_spy],
    axis=1,
    join='inner'
)


def main():
    # Part 1: Lagged OLS regression
    lag_days = 730  # two-year lag
    df_reg = daily.copy()
    df_reg['box_lag'] = df_reg['box_office'].shift(lag_days)
    df_reg.dropna(subset=['netflix', 'box_lag'], inplace=True)

    X = add_constant(df_reg['box_lag'])
    y = df_reg['netflix']
    model = OLS(y, X).fit()
    print('\nLagged Regression Results (Netflix ~ BoxOffice lagged by 730 days):')
    print(model.summary())

    # Part 2: Cross-correlation plot (±365 days)
    maxlag = 365
    lags = np.arange(-maxlag, maxlag + 1)
    corrs = [
        daily['box_office'].corr(daily['netflix'].shift(l))
        for l in lags
    ]

    plt.figure()
    plt.plot(lags, corrs)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Lag (days; positive means BoxOffice leads)')
    plt.ylabel('Correlation')
    plt.title('Box Office vs Netflix Cross-Correlation (±365 days)')
    plt.tight_layout()
    plt.savefig('outputs/box_net_corr.png')
    plt.close()
    print('\nSaved cross-correlation plot to outputs/box_net_corr.png')

    # Part 3: Granger causality tests (BoxOffice → Netflix)
    print('\nGranger Causality Tests: does BoxOffice lead Netflix?')
    gc_df = daily[['netflix', 'box_office']]
    results = grangercausalitytests(gc_df, maxlag=8, verbose=False)
    for lag, res in results.items():
        pval = res[0]['ssr_ftest'][1]
        print(f"  Lag {lag:>2}: p-value = {pval:.4f}")


if __name__ == '__main__':
    main()
