# Applied_Machine_Learning
AML assignment

## Objective
Make profit using the following trading strategy:

Buy at open price and sell at close if Close is higher than open 

### Success Metric
1. Percent profit in 1 week
2. Percent of trades lost (False positives)


## ML Approach
To build a binary classifier using OHLC data from yfinance to predict if open price is lower than close price for tomorrow

### Performance metrics
1. Accuracy
2. Precision
3. Recall

### Features
1. Previous day OHLC
2. Previous day Volume
3. Previous day [Close - Open]
4. Previous day [Close-Open]* Volume
5. Last week average closing price

### Target
Is Close > Open ?

## Benchmarks
1. Say yes to everything
2. Random Guess with 50-50 probability
3. Regress on (close - open) or both seperately
4. Logistic regression model on last week's moving average of Close + previous day OHLC

