# Portfolio Construction based on Predicted Stock Returns and Volatilities


## Abstract

We seek to provide a useful and easy-to-use algorithm for common investors to construct a portfolio that considers both historical stock performance and short-term market sentiment signals. This algorithm selects stocks from a selection of S&P 500 stocks with negative alphas, and calculates the stocks’ volatility and expected returns using both historical data and current market sentiment. With these adjusted expected returns and volatility, the algorithm produces an optimized portfolio based economic theories.

See *YNR.ipynb* for detailed model establishment and explanation
See *Volatility Sentiment Analysis.ipynb* for sentiment analysis and justification of exclusion of Reddit posts and Tipranks trading information in the final algorithm
See *portfolioConstruction.py* with functions only for direct portfolio construction


## User Usage

File:  portfolioConstruction.py

Main Function: `find_optimal_port(threshold = 0.01, date = dt.datetime.now())`

Input: 
1. `threshold`: an integer representing the maximum volatility the user can accpet, default to be 0.01(1%)
2. `date`: the time the portfolio construction is based on, default to be today

Output: a dataframe that reflects the optimized portfolio, with predicted returns and volatility, as well as weights of the stocks

Example:
```python
find_optimal_port()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Returns</th>
      <th>Volatility</th>
      <th>VRTX weight</th>
      <th>CAG weight</th>
      <th>CPB weight</th>
      <th>PRGO weight</th>
      <th>VRSN weight</th>
      <th>EQR weight</th>
      <th>NOV weight</th>
      <th>V weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
    <th>24</th>
    <th>0.004875</th>
    <th>0.009979</th>
    <th>0.066031</th>
    <th>0.125143</th>
    <th>0.249491</th>
    <th>0.122638</th>
    <th>0.252543</th>
    <th>0.138797</th>
    <th>0.014432</th>
    <th>0.030924</th>
    </tr>
  </tbody>
</table>
</div>



## Documentation

`seek_alpha()`
- Output: a `dictionary` containing tickers of stocks with postivie, neutral and negative alphas in the time period 2020.5 to 2021.5

- Example: 
```python
D = seek_alpha()
pos, neg, obj = D["Positive"], D["Negative"], D["Neutral"]

# first 5 stocks in the list of stocks with negative alphas,
# and the number of stocks with negative alphas
neg[:5], len(neg)
```
```
(['BAM', 'ICE', 'TOT', 'EPAY', 'TMQ'], 110)
```
----------------

`get_tweets(tickers, begin, end)`
- Input:
  - tickers: a `list` of tickers of stocks about which to scrape for tweets
  - begin: the begin date of the time period in which tweets are scraped, `dt.datetime` object
  - end: the end date of the time period in which tweets are scraped, `dt.datetime` object

- Output:
A `dataframe` containing the Tweets, Date, and Ticker in its columns

- Example: 
```python
get_tweets(["AAPL","TSLA"], dt.datetime(2021,4,20), dt.datetime(2021,4,21))
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
      <th>Date</th>
      <th>Ticker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <th>$AAPL NEW ARTICLE : Apple event fails to save ... </th>
      <th>2021-04-20</th>
      <th>AAPL</th>
    </tr>
    <tr>
      <th>1</th>
      <th>@Stash COME JOIN THE PARTY #StockStashParty an...</th>
      <th>2021-04-21</th>
      <th>AAPL</th>
    </tr>
    <tr>
      <th>2</th>
      <th>✨ Participate in the April #Webull wheel event...</th>
      <th>2021-04-20.</th>
      <th>TSLA</th>
    </tr>
    <tr>
      <th>3</th>
      <th>Don’t tell Tommy but a whale has been buying 🐳...</th>
      <th>2021-04-21</th>
      <th>TSLA</th>
    </tr>
  </tbody>
</table>
</div>

----------------

`get_prices(tickers, begin, end):`
- Input:
- - tickers: a `list` of tickers of stocks of which to retrieve returns
- - begin: the begin date of the time period in which returns are retrieved, `dt.datetime` object
- - end: the end date of the time period in which returns are retrieved `dt.datetime` object

- Output:
A `dataframe` containing daily price changes for each specified stock in the specified time period


- Example: 
```python
get_prices(['AAPL','TSLA'], dt.datetime(2021,1,1), dt.datetime(2021,4,30))
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>AAPL</th>
      <th>TSLA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <th>2021-01-04</th>
      <th>-0.024719</th>
      <th>0.034152</th>
    </tr>
    <tr>
      <th>1</th>
      <th>2021-01-05</th>
      <th>0.012364</th>
      <th>0.007317</th>
    </tr>
    <tr>
      <th>2</th>
      <th>2021-01-06</th>
      <th>-0.033662</th>
      <th>0.028390</th>
    </tr>
    <tr>
      <th>3</th>
      <th>2021-01-07</th>
      <th>0.034123</th>
      <th>0.079447</th>
    </tr>
    <tr>
      <th>4</th>
      <th>2021-01-08</th>
      <th>0.008631</th>
      <th>0.078403</th>
    </tr>
  </tbody>
</table>
</div>

----------------

`predict_return(ticker, date = dt.datetime.now())`
- Input:
- - ticker: a `string` of one stock ticker
- - date: the day for which the return is predicted, `dt.datetime` object, dafault to be today

- Output:
A `numerical` predicted return of the input stock

----------------

`predict_returns(tickers, date = dt.datetime.now())`
- Input:
- - tickers: a `list` of tickers of stocks of which to retrieve returns
- - date: the day for which the returns are predicted, `dt.datetime` object, dafault to be today

- Output:
A `dataframe` containing tickers and their corresponding predicted returns

----------------

`stock_var(tickers, date = dt.datetime.now())`
- Input:
- - tickers: a `list` of tickers of stocks of which to retrieve returns
- - date: the day for which the returns are predicted, `dt.datetime` object, dafault to be today

- Output: a `dataframe` representing the variance covariance matrix for the specified stocks in a one-year period ending at the specified date

----------------

`save_sp500_tickers()`
- Output: a `list` of all sS&P500 tickers
