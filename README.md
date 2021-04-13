# Project Proposal


## Abstract

We seek to provide a useful and easy-to-use algorithm for common investors to construct a portfolio that considers both historical stock performance and short-term market sentiment signals. This algorithm selects stocks (and possibly their derivatives) from a selection specified by the user, and calculates the stocks’ volatility and expected returns using both historical data and current market sentiment. With these adjusted expected returns and volatility, the algorithm produces an optimized portfolio based economic theories.
 
## Planned Deliverables

The algorithm will be wrapped up as a function (or potentially python package) which allows the user to directly utilize the function through simply supplying arguments. 

- Full Success: the project produces an algorithm that constructs optimized portfolios which historically performs better than the given benchmark (SPX500 / NASDAQ100) within a given period of time
- Partial Success: 1) The project might only be able to calculate and predict volatility and expected returns without producing a portfolio 2) Due to the lack of access to market information, the project may only operate on a limited set of stocks and financial tools 3) It’s able to produce an optimized portfolio but such portfolio performs poorly in retrospective testing

## Resources Required
 
The majority parts of the project will rely on open-source data:
 
- 1) Financial Data extracted from Yahoo Finance (which can be easily accessed through yfinance python module) and 
- 2) the sentiment analysis part rely on other publicly available comments on the Internet. (which may include review by analysts, hotspots on Reddit, financial news coverage, etc.) 
 
While unlikely, the project may also require information about financial derivatives (e.g. Options, Futures, Bonds) Some of those might require access to Bloomberg or other professional databases. It should be noted that the project will be able to function well without these add-ons, but we might experiment with some of them in order to achieve a higher rate of returns and lower level of volatility.
