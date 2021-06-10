# -*- coding: utf-8 -*-

# Import Packages

import pandas_datareader.data as reader
import datetime as dt
import pandas as pd
import numpy as np
import urllib, json
import httplib2

import snscrape.modules.twitter as sntwitter

import tensorflow as tf
import re
import string

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras
from bs4 import BeautifulSoup

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt

import plotly.express as px

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import pydot
import graphviz

import bs4 as bs
import pickle
import requests

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
tf.autograph.set_verbosity(1)

# Preparation Functions

def seek_alpha():
    # Read in the processed data
    stocks = pd.read_csv("US_cleaned.csv")
    market = pd.read_csv("SPX_cleaned.csv")
    
    # Convert the Data column to dt.datettime object
    stocks['Date'] = pd.to_datetime(stocks['Date'], format='%Y/%m/%d')
    market['Date'] = pd.to_datetime(market['Date'], format='%Y/%m/%d')  
    
    # Using only data from 2020.5.1 onwards (in the past 12 months)
    stocks = stocks[stocks['Date'] >= "2020/5/1"].copy().reset_index(drop = True)
    market = market[market['Date'] >= "2020/5/1"].copy().reset_index(drop = True)
    
    # Append the market return to the stock data dataframe
    three_factor_past = pd.concat([stocks,pd.concat([market]*int(len(stocks)/12)).reset_index()[['Market']]],axis = 1)
    
    # Pull out the data in 2020.5 to determine the groups of the stocks
    det  = three_factor_past.loc[three_factor_past["Date"] == dt.datetime(2020,5,1)].copy()
    det = split_MK(split_PB(det))
    det["Group"] =det["group_MK"]  + "/" +  det["group_PB"] 
    
    # Append the group assignments back to the original dataframe
    three_factor_past = pd.merge(three_factor_past,det.drop(["Date","PB","MK","Market","Return"],axis=1),
                             how="outer",left_on=["Ticker"],right_on=["Ticker"])
    
    # Calculate the average return in each group for each of the 12 months
    factor_return = three_factor_past.groupby(['Date','Group']).apply(lambda x: (x['Return']*x['MK']).sum()/x.MK.sum())
    factor_return = factor_return.reset_index()
    factor_return.columns = ["Date","Group","Return"]
    factor_return = factor_return.pivot(index= "Date", columns= "Group", values= "Return")
    
    # Calculate the premiums in the market for each month
    factor_return = SMB(HML(factor_return))
    
    # Extract the premiums
    f = factor_return.loc[:,["SMB","HML"]].copy()
    f.columns=['SMB','HML']
    
    # Append the premiums to each ticker
    three_factor_past = pd.concat([three_factor_past,pd.concat([f]*int(len(three_factor_past)/12)).reset_index()[["SMB","HML"]]],axis = 1)
    
    # Set the indicises and add an constant term to perform regression
    df = three_factor_past.set_index(keys = ["Ticker","Date"])
    df["Intercept"] = 1
    
    # Prepare the ticker list for iterated OLS estimates
    tickers = list(set(three_factor_past["Ticker"]))
    # Initialize the containers for different alpha groups
    alpha_pos = []
    alpha_neg = []
    alpha_obj = []
    
    # For every ticker in the ticker list
    for ticker in tickers:
        # Extract the stock data correponding to the ticker
        T = df.loc[ticker].reset_index()
        # Estimate the OLS model
        model = sm.OLS(T["Return"], T[["SMB","HML","Market","Intercept"]])
        results = model.fit()
        # If the alpha value is significant for the stock
        if results.pvalues[3] < 0.1:
            # If alpha is positive
            if results.params[3] > 0:
                # Assign the ticker to the alpha positive group
                alpha_pos.append(ticker)
            # If alpha is negative
            else:
                # Assign the ticker to the alpha negative group
                alpha_neg.append(ticker)
        # If alpha is not significant
        else:
            # Assign the ticker to the alpha neutral group
            alpha_obj.append(ticker)
    
    # Concentrate the result into a dictionary and return
    D = {"Positive": alpha_pos, "Neutral": alpha_obj, "Negative": alpha_neg}
  
    return D

def save_sp500_tickers():
    """
    A function that retrieves the S&P 500 Tickers from the Wikipedia page
    
    Output
    --------
    A list containing all S&P500 tickers
    """
    # Using bs to first download the corresponding webpage
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    # Initialize the container
    tickers = []
    # Scrape all tickers
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    
    # Strip all redundant suffix
    tickers = list(map(lambda s: s.strip(), tickers))

    return tickers

def get_tweets(tickers, begin, end):
    """
    A function retrieving tweets between a specified peiod of time. For each given day, only 20000 characters of tweets will be retrieved at maximum.
    
    Input
    --------
    tickers: list of tickers the user want to scrape for tweets
    begin: dt.datetime object, the beginning of the time period
    end: dt.datetime object, the end of the time period
    
    Output
    --------
    A dataframe containing the tweets, tikcers and dates
    """
    # Initialize the container
    data = []
    # For all tickers 
    for ticker in tickers:
        # Initialize a tracker for the while loop
        date = begin
        # For all days in the time period
        while date < end + dt.timedelta(days = 1):
            # Set the time mark for that specific day
            d1 = date
            d2 = date + dt.timedelta(days = 1)
            # Container for tweets
            tweets = ''
            # Tracking the length of tweet
            length = 0
            # Retrieve relevant tweets on that speicifc day
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{ticker} since:{d1.year}-{d1.month}-{d1.day} until:{d2.year}-{d2.month}-{d2.day}').get_items()):
                # Exclude those tweets if the ticker appears in the userid
                if ticker not in tweet.username.lower():
                    # Add the tweets to the container
                    tweets += tweet.content
                    # Update the length
                    length += len(tweet.content)
                # If we have reached our targeted length
                if length > 20000:
                    # Exit the loop
                    break
            # Construct a dataframe containing the tweets, the ticker and the date
            df = pd.DataFrame({"Tweets": [tweets]})
            df["Date"] = date
            df["Ticker"] = ticker
            
            # Append the dataframe to the container
            data.append(df)
            # Update the date
            date += dt.timedelta(days = 1)
    # Convert the container to a mega dataframe and return
    result = pd.concat(data, ignore_index = True)

    return result

def get_prices(tickers, begin, end):
    """
    A function retrieving daily price changes between a specified peiod of time using Yahoo Finance.
    
    Input
    --------
    tickers: list of tickers the user want to retrieve returns for
    begin: dt.datetime object, the beginning of the time period
    end: dt.datetime object, the end of the time period
    
    Output
    --------
    A dataframe containing the daily price change, tikcers and dates
    """
    # Initialize the container
    data = []
    # For all tickers
    for ticker in tickers:
        # Retieve the daily price using the yahoo finance module
        de = reader.get_data_yahoo(ticker,begin,end)[['Adj Close']]
        data.append(de)
    # Concat the data into a single dataframe
    result = pd.concat(data, axis = 1)
    # Set the names of the columns to be the tickers
    result.columns = tickers
    # Calculate the daily price changes reset the indices
    result = result.pct_change().dropna().reset_index()

    return result

def lag(x, num_places):
    """
    Shifts the array x by num_places positions to the right,
    Fills positions now empty with np.nan values,
    Returns the shifted array.
    
    arguments:
    x -> 1d array of float types
    num_places -> int type, number of positions to shift the array
    
    return:
    y -> 1d array after the shift
    """
    # Shift the array by num_places positions
    new = np.roll(x, num_places)
    # Fill the emptied positions with np.nan
    new[0:num_places] = np.nan
    
    # Return the shifted array
    return new

def tweets_input(ticker, begin, end):
    """
    A function that returns a lagged version of tweets input for a specific ticker over a certain period.
    
    Input:
    ticker: string, the stock ticker
    begin: dt.datetime object, the beginning of the time period
    end: dt.datetime object, the end of the time period
    
    Output:
    A dataframe containing a lagged version of tweets input for a specific ticker over a certain period
    """
    # Retrieve all tweets in the given period of time
    df = get_tweets([ticker], begin - dt.timedelta(days = 2), end - dt.timedelta(days = 1))
    # Note that we didn't retrieve the tweets for the lat day, in order for the lag function to consider the last day
    # We add an extra row
    last_day = pd.DataFrame({"Date": end,"Tweets": '',"Ticker": ticker}, index=[0])
    df = pd.concat([df,last_day], ignore_index = True)
    # Do the lag for two days
    for i in range(1,3):
        df["Tweets " + str(i) + " Days Before"] = df.groupby("Ticker")["Tweets"].transform(lambda x: lag(x, i))
    # Drop the tweets for the given day
    df = df.drop("Tweets", axis = 1)
    # Dropna and reset the indices
    df = df.dropna().reset_index(drop = True)
    
    return df

def return_input(ticker, begin, end):
    """
    A function that returns a lagged version of daily price changes input for a specific ticker over a certain period.
    
    Input
    --------
    ticker: string, the stock ticker
    begin: dt.datetime object, the beginning of the time period
    end: dt.datetime object, the end of the time period
    
    Output
    --------
    A dataframe containing a lagged version of daily price changes input for a specific ticker over a certain period
    """
    # Retrieve the daily returns for all tradedays in the period, note that there are holidays and weekends, we handle this by retrieving
    # also data further away before the beginning of the time period and drop them at the end
    df = get_prices([ticker], begin - dt.timedelta(days = 10), end - dt.timedelta(days = 1))
    df["Ticker"] = ticker
    df = df.rename(columns = {ticker: "Return"})
    # Note that we didn't retrieve the returns for the lat day, in order for the lag function to consider the last day
    # We add an extra row
    last_day = pd.DataFrame({"Date": end,"Return":np.NaN, "Ticker": ticker}, index=[0])
    df = pd.concat([df,last_day], ignore_index = True)
    # Do the lag for 5 days
    for i in range(1,6):
        df["Return " + str(i) + " Days Before"] = df.groupby("Ticker")["Return"].transform(lambda x: lag(x, i))
    # Drop the data before the beginning of the time period
    df = df[df["Date"] >= begin].copy().reset_index(drop = True)

    return df

def identify_trade_days(x, dates):
    """
    A function to mark the dataframe rows if the date is within the given list of tradedays
    
    Input
    --------
    x: DataFrame object grouped by Date
    dates: list of tradedays as dt.datetime object
    
    Output
    --------
    The dataframe with rows marked, 1 in the TD column means the row is data from a tradeday and 0 otherwise
    """
    # If the date is a tradeday
    if x["Date"].values[0] in dates:
        # Mark the row accordingly
        x["TD"] = 1
    # If the date is not a tradeday
    else:
        # Mark the row acoordingly
        x["TD"] = 0

    return x

def select_trade_days(x, dates):
    """
    A function that selects all the rows in the DataFrame whose date is within the given list of tradedays
    
    Input
    --------
    x: DataFrame object grouped by Date
    dates: list of tradedays as dt.datetime object
    
    Output
    --------
    The part of the dataframe whose dates are tradedays
    """
    # Mark the rows to identify the trade days
    df = x.groupby("Date").apply(identify_trade_days, dates = dates)
    # Drop all the non-tradedays
    df = df[df["TD"] == 1].copy().reset_index(drop = True)
    # Return the result

    return df.drop("TD", axis = 1)

def remove_stopwords(df_pd, colname, stopwords_list):
    """
    The function looks at the column df_pd[colname] in the input dataframe df_pd,
    and first change all characters to lowercase, 
    split texts into a list of word chunks according to empty space,
    and then reconcatenate the word chunks that are not in the list of stopwords stopwords_list
    into a string of texts again.

    Limitation: this function does not deal with stopwords right before punctuation marks.

    Input
    --------
    df_pd: a pandas dataframe containing text data
    colname: the name of the dataframe column that has text data
    stopwords_list: a list of stopwords, to be removed in the target text

    Output
    --------
    a pandas dataframe with the text with stopwords removed in the target column
    """
    all_words_in_list = df_pd[colname].str.lower().str.split()
    stopwords_removed = all_words_in_list.apply(lambda x: ' '.join([word for word in x if word.lower() not in stopwords_list]))

    return stopwords_removed

def make_tensorflow_dataset(ticker, begin, end):
    
    tweets = tweets_input(ticker, begin, end)
    returns = return_input(ticker, begin, end + dt.timedelta(days = 1)).dropna()
    
    dates = returns["Date"].tolist()
    tweets = select_trade_days(tweets, dates)
    
    stopwords_list = stopwords.words("english")  # a list of stopwords in english
    tweets["Tweets 1 Days Before"] = remove_stopwords(tweets, "Tweets 1 Days Before", stopwords_list)
    tweets["Tweets 2 Days Before"] = remove_stopwords(tweets, "Tweets 2 Days Before", stopwords_list)

    df = pd.merge(returns, tweets)
    columns = ["Return " + str(i) + " Days Before" for i in range(1,6)]
    
    data = tf.data.Dataset.from_tensor_slices(
        (
            {
                "Tweets_1" : df[["Tweets 1 Days Before"]], 
                "Tweets_2" : df[["Tweets 2 Days Before"]], 
                "Previous_Returns": df[columns]
            }, 
            {
                "Returns" : df[["Return"]]
            }
        )
    )

    return data

def predict_return(ticker, date = dt.datetime.now()):
    # We are going to train the model based on the behavior in the past month
    date = dt.datetime(date.year, date.month, date.day)
    begin = date - dt.timedelta(days = 31)
    end = date - dt.timedelta(days = 1)
    
    # Retrieve the relevant data
    data = make_tensorflow_dataset(ticker, begin, end)
    
    # Define a vocab cap for vectorization
    size_vocabulary = 2000

    # Vectorize the tweets
    vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 
    vectorize_layer.adapt(data.map(lambda x, y: x["Tweets_1"]))
    vectorize_layer.adapt(data.map(lambda x, y: x["Tweets_2"]))
    
    # Contruct the tensorflow inputs
    tw_input_1 = keras.Input(
        shape = (1,), 
        name = "Tweets_1",
        dtype = "string"
    )

    tw_input_2 = keras.Input(
        shape = (1,), 
        name = "Tweets_2",
        dtype = "string"
    )

    re_input = keras.Input(
        shape = (5,), 
        name = "Previous_Returns",
        dtype = "float64"
    )
    
    # Create a common embedding layer
    embedding_layer = layers.Embedding(input_dim = size_vocabulary, output_dim = 10)

   # vectorize title texts into tensor containing data
    tw_features1 = vectorize_layer(tw_input_1)
    # Turns positive integers (indexes) into dense vectors of fixed size
    tw_features1 = embedding_layer(tw_features1)
    # Applies Dropout to the input to avoid overfitting
    tw_features1 = layers.Dropout(rate = 0.2)(tw_features1)
    # pooling To reduce variance, reduce computation complexity
    tw_features1 = layers.GlobalAveragePooling1D()(tw_features1)
    # Applies Dropout to the input to avoid overfitting
    tw_features1 = layers.Dropout(rate = 0.2)(tw_features1)
    # add a dense layer
    tw_features1 = layers.Dense(32, activation = "relu", name = "tw_dense1")(tw_features1)

    # vectorize title texts into tensor containing data
    tw_features2 = vectorize_layer(tw_input_2)
    # Turns positive integers (indexes) into dense vectors of fixed size
    tw_features2 = embedding_layer(tw_features2)
    # Applies Dropout to the input to avoid overfitting
    tw_features2 = layers.Dropout(rate = 0.2)(tw_features2)
    # pooling To reduce variance, reduce computation complexity
    tw_features2 = layers.GlobalAveragePooling1D()(tw_features2)
    # Applies Dropout to the input to avoid overfitting
    tw_features2 = layers.Dropout(rate = 0.2)(tw_features2)
    # add a dense layer
    tw_features2 = layers.Dense(32, activation = "relu", name = "tw_dense2")(tw_features2)
    
    # Combine the two tweets feature sub-model
    tw_features = layers.concatenate([tw_features1, tw_features2], axis = 1)
    tw_features = layers.Dense(32, activation='relu')(tw_features)

   # Build a return feature model
    re_features = layers.Dense(32, activation='relu')(tf.reshape(re_input, [-1, 5]))
    re_features = layers.Dense(16, activation='relu')(re_features)

    # concatenate and create output
    main = layers.concatenate([tw_features1, tw_features2, re_features], axis = 1)
    main = layers.Dense(32, activation='relu')(main)
    output = layers.Dense(1, name = "Returns")(main)
    
    # Declare the model
    combined_model = keras.Model(
        inputs = [tw_input_1, tw_input_2, re_input],
        outputs = output
    )
    
    #complie
    combined_model.compile(optimizer = "adam",
                  loss = losses.MeanSquaredError(),
                  metrics = keras.metrics.MeanSquaredError()
    )

    # train
    combined_history = combined_model.fit(data, 
                        epochs = 30,
                        verbose = False)
    
    # Make the prediction on the specific day
    prediction_input = get_prediction_tensorflow_input(ticker, date)
    result = combined_model.predict(prediction_input)[0][0]
    
    return result

def predict_returns(tickers, date = dt.datetime.now()):
    returns = []
    for ticker in tickers:
        re = predict_return(ticker, date)
        returns.append(re)
    df = pd.DataFrame({"Ticker": tickers, "Returns": returns})
    return df

def find_optimal_port(threshold = 0.01, date = dt.datetime.now()):
    """
    A wrap up function that produces the optimized portfolio using long-term stock selection and short-term return prediction
    
    Input
    --------
    threshold: int, the maximum volatility that the user is willing to stand, default at 1%
    date: the time for which the portfolio is produced, default set to today
    
    Output
    --------
    A dataframe representing the optimized portfolio with specified weights for each stock, the predicted return and volatility
    """
    
    # Find the stocks with negative alphas
    D = seek_alpha()
    neg =  D["Negative"]
    
    # Select those in the S&P 500
    SP500 = save_sp500_tickers()
    neg = [i for i in neg if i in SP500]
    
    # Predict the returns
    re = predict_returns(neg, date)
    
    # Select stocks with positive returns
    re = re[re["Returns"] > 0]
    tk = re.Ticker
    
    # Retrieve the variance covariance matrix
    vo = stock_var(tk, date)
    
    # Find the optimal portfolio
    p = simulate_portfolios(re, vo)
    opt_p = get_optimal(p, threshold).reset_index(drop = True)

    return opt_p

def stock_var(tickers, date = dt.datetime.now()):
    """
    A function to retrieve the variance covariance matrix for the specified stocks on a given day using data in one-year period before that day
    
    Input:
    tickers: list of stock tickers
    date: dt.datetime object, the day for which the variance covariance matrix will be produced
    """
    # Standardize the datetime object for yahoo finance data reader
    date = dt.datetime(date.year, date.month, date.day)
    # Read in the price change over the past year
    df = reader.get_data_yahoo(tickers, date - dt.timedelta(days = 365), date - dt.timedelta(days = 1))["Adj Close"]
    # Produce the variance covariance matrix
    result = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    
    return result

def simulate_portfolios(re, vo):
    """
    A portfolio simulation function that generates 100,000 portfolios using the given returns and variance covariance matrix.
    
    Input:
    re: DataFrame containing the predicted returns
    vo: DataFrame containing the variance covariance matrix
    
    Output:
    A DataFrame containing 100,000 portfolios, the weights in each portfolio and the its expected return and volatility
    """
    # Initialize the containers
    portfolio_re = []
    portfolio_vo = []
    portfolio_weights = []
    
    # Obtain the number of stocks we are looking at
    num_instru = len(re)
    # Create 100,000 portfolios
    for i in range(100000):
        # Assign random weights
        weights = np.random.random(num_instru)
        weights = weights/np.sum(weights)
        portfolio_weights.append(weights)
        
        # Calculate the returns
        returns = weights @ re["Returns"]
        portfolio_re.append(returns)
        
        # Calculate the volatility
        var = vo.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        v = np.sqrt(var)
        portfolio_vo.append(v)
    
    # Combine the results
    df = pd.DataFrame({'Returns':portfolio_re, 'Volatility':portfolio_vo})
    for i in range(num_instru):
        df[re.iloc[i,0] + ' weight'] = [w[i] for w in portfolio_weights]
    
    return df

def get_optimal(p, threshold):
    """
    A function that retrieves the optimal portfolio (i.e. the portfolio with the highest rate of return under the volatility threshold
    passed by the user)
    
    Input:
    p: DataFrame containing all portfolios
    threshold: float, the volatility threshold passed by the user
    
    Output:
    A DataFrame containing the optimal portfolio with weights of each stock in the portfolio
    """
    # Select the portfolios with volatility under the threshold
    p_1 = p[p["Volatility"] <= threshold].copy().reset_index(drop = True)
    # Find the optimal portfolio
    op_port = p_1.iloc[p_1["Returns"].idxmax()]
    # Construct a DataFrame
    df = pd.DataFrame(op_port).T
    
    # Return the portfolio
    return df

# Final Algorithm for Direct Use

def find_optimal_port(threshold = 0.01, date = dt.datetime.now()):
    """
    A wrap up function that produces the optimized portfolio using long-term stock selection and short-term return prediction
    
    Input:
    threshold: int, the maximum volatility that the user is willing to stand, default at 1%
    date: the time for which the portfolio is produced, default set to today
    
    Output:
    A dataframe representing the optimized portfolio with specified weights for each stock, the predicted return and volatility
    """
    
    # Find the stocks with negative alphas
    D = seek_alpha()
    neg =  D["Negative"]
    
    # Select those in the S&P 500
    SP500 = save_sp500_tickers()
    neg = [i for i in neg if i in SP500]
    
    # Predict the returns
    re = predict_returns(neg, date)
    
    # Select stocks with positive returns
    re = re[re["Returns"] > 0]
    tk = re.Ticker
    
    # Retrieve the variance covariance matrix
    vo = stock_var(tk, date)
    
    # Find the optimal portfolio
    p = simulate_portfolios(re, vo)
    opt_p = get_optimal(p, threshold).reset_index(drop = True)
    
    return opt_p
