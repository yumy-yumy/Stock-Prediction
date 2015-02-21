import Quandl
from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def getData(name):
    stockList = dict([("google", "GOOG/NASDAQ_GOOG")])
    url = stockList[name]
    stock = Quandl.get(url, trim_start="2014-03-27", trim_end='2015-02-13', authtoken="W1ByxzyeTjYzXSVtACKV")
   
    return stock 


def getTrainingAndTestingData(stock, win, precent):
    N = len(stock)
    closePrice = np.array(stock.Close)
    highPrice = np.array(stock.High)
    lowPrice = np.array(stock.Low)
    volume = np.array(stock.Volume)
    # input X
    X = pd.DataFrame()
    
    X['RSI'] = RSI(closePrice, win)
    
    '''
    bollingerUp, bollingerLow = SMA(closePrice, win)
    bollinger = pd.DataFrame(data=zip(bollingerUp, bollingerLow), columns=['BollingerUp', 'BollingerLow'])
    X[['BollingerUp', 'BollingerLow']] = bollinger[['BollingerUp', 'BollingerLow']]
    '''
    X['dfHighLow'] = diffHighLow(closePrice, highPrice, lowPrice)
    '''
    X['dfClose'] = diff(closePrice)
    X['dfMinClose'] = diffMin(closePrice)
    X['dfMaxClose'] = diffMax(closePrice)
    '''
    X['dfVoume'] = diff(volume)
    X['dfMinVoume'] = diffMin(volume)
    X['dfMaxVoume'] = diffMax(volume) 
    '''
    X['K'] = K(closePrice, highPrice, lowPrice, win)
    X['D'] = D(X['K'])
    '''
    # label y (next-day prediction)
    y = closePrice[2:] - closePrice[1:-1]
    y = y >= 0
    
    '''
    # label y (long-term prediction)
     y = closePrice[long:] - closePrice[:-long]
    y = y >= 0
    '''
    trainNum = int(N*precent)
    
    return X[:trainNum], y[:trainNum], X[trainNum:], y[trainNum:]

def svmClassifier(train_X, train_y, test_X, test_y):
    clf = svm.SVC()
    clf.fit(train_X, train_y)
    prediction = clf.predict(test_X)
    correct = np.count_nonzero(prediction==test_y)
    accuracy = float(correct)/len(prediction)
    
    return accuracy

# relative strength index(RSI)
def RSI(price, win):
    N = len(price)
    df = price[1:] - price[:-1]
    
    gain = np.copy(df)
    np.place(gain, gain<0, 0)  
    lost = np.copy(df)
    np.place(lost, lost>0, 0)
    lost = np.abs(lost)
    
    cumGain = list()
    cumLost = list()
    for i in range(1, N-1):
        if i < win-1:
            cumGain.append(np.sum(gain[:i]))
            cumLost.append(np.sum(lost[:i]))            
        else:
            cumGain.append(np.sum(gain[i-win+1:i]))
            cumLost.append(np.sum(lost[i-win+1:i]))
    cumGain = np.array(cumGain)
    cumLost = np.array(cumLost)
    rs = cumGain/cumLost
    rsi = 100 - 100/(1+rs)

    return rsi

'''
 bollinger upper band(simple moving average in the middle plus standard deviation), 
 bollinger lower band(simple moving average in the middle minus standard deviation)
'''
def SMA(price, win):
    N = len(price)
    
    bollinger_up = list()
    bollinger_low = list()
    for i in range(2, N):
        if i < win:
            sma = np.mean(price[:i])
            std = np.std(price[:i])       
        else:
            sma = np.mean(price[i-win:i])
            std = np.std(price[i-win:i])
        bollinger_up.append(sma+std)
        bollinger_low.append(sma-std)
    
    bollinger_up = np.array(bollinger_up)
    bollinger_low = np.array(bollinger_low)
    
    price = price[1:-1]
    bollinger_up = (price - bollinger_up) / bollinger_up
    bollinger_low = (price - bollinger_low) / bollinger_low
 
    return bollinger_up, bollinger_low

def diffHighLow(close, high, low):
    close = close[1:-1]
    high = high[1:-1]
    low = low[1:-1]
    df = (close - low) / (high - low)
    return df

# difference between today and yesterday
def diff(item):
    item = item[1:]
    df = (item[1:] - item[:-1])/item[:-1]
    return df
    
# difference between today and min of a period
def diffMin(item):
    period = 5
    N = len(item)
    df = list()
    for i in range(2, N):
        if i < period:
            df.append((item[i-1] - np.min(item[:i])) / np.min(item[:i]))
        else:
            df.append((item[i-1] - np.min(item[i-period:i])) / np.min(item[i-period:i]))
    
    return df

# difference between today and max of a period
def diffMax(item):
    period = 5
    N = len(item)
    df = list()
    for i in range(2, N):
        if i < period:
            df.append((item[i-1] - np.max(item[:i])) / np.max(item[:i]))
        else:
            df.append((item[i-1] - np.max(item[i-period:i])) / np.max(item[i-period:i]))
    
    return df 

# stochastic %K
def K(close, high, low, win):
    N = len(close)
    k = list()
    for i in range(2, N):
        if i < win:
            k.append(100*(close[i-1]-np.min(low[:i])/(np.max(high[:i])-np.min(low[:i]))))
        else:
            k.append(100*(close[i-1]-np.min(low[i-win:i])/(np.max(high[i-win:i])-np.min(low[i-win:i]))))
    
    return k

# 3-period moving average of %K
def D(k):
    period = 3
    N = len(k)
    d = list()
    for i in range(1, N+1):
        if i < period:
            d.append(np.mean(k[:i]))
        else:
            d.append(np.mean(k[i-period:i]))
    
    return d

'''
    find suitable sliding window size by a walk-forward testing model
'''
def walkForwardTesting(stock, percent):    
    windows = range(2, 10)
    # 3 months
    period = 3
    # 30 days
    size = 30
    accuracyList = dict()
    
    for win in windows:
        accuracy = 0
        train_X, train_y, test_X, test_y = getTrainingAndTestingData(stock, win, percent)
        N = len(train_X)/size
        for i in range(0, N-period):
            accuracy += svmClassifier(train_X[i*size:(i+period)*size],
            train_y[i*size:(i+period)*size], train_X[(i+period)*size:(i+period+1)*size],
            train_y[(i+period)*size:(i+period+1)*size])
        i += 1   
        if len(train_X)%size != 0:
            accuracy += svmClassifier(train_X[i*size:(i+period)*size],
            train_y[i*size:(i+period)*size], train_X[(i+period)*size:],
            train_y[(i+period)*size:])
        accuracyList[win] = accuracy
    
    win = max(accuracyList, key=accuracyList.get)
        
    return win

'''
    L1-based eature selection

def featureSelection():
    clf = Pipeline([
                    ('feature_selection', ),
                    ('classification', svm.SVC())])
    return
'''


if __name__ == "__main__":
    
    stock = getData("google")
    
    win = 2
    
    train_X, train_y, test_X, test_y = getTrainingAndTestingData(stock, win, 0.8)
    
    res = svmClassifier(train_X, train_y, test_X, test_y)
    
    print res
    '''
    
    bestWin = walkForwardTesting(stock, 0.7)
    
    print bestWin
    '''
    

    
