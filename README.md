# Stock Prediction using Long short-term memory (LSTM)
Implementation of LSTM to predict stock price.

The implementation use reference from this article, https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb

The stock data from https://finance.yahoo.com/

## Note
This project using data from yahoo finance, and implemented to predict Close Price stock from :
- Apple (AAPL)
- Dow Jones Industrial Average (^DJI)
- Hang Seng Index (^HSI)
- S&P 500 (^GSPC)

##
Preview

![S&P 500 stock prediction](https://github.com/kokohi28/stock-prediction/blob/master/GSPC_sample.png?raw=true)


## Requirements
* Python 3.7

## Requirements Library
* numpy ->
  $ pip install pip install numpy

* pandas ->
  $ pip install pandas

* pandas_datareader ->
  $ pip install pandas-datareader

* sklearn ->
  $ pip install scikit-learn

* keras ->
  $ pip install Keras

* tensorflow ->
  $ pip install tensorflow

* matplotlib ->
  $ pip install matplotlib

## File Structure
### py files
* const.py -> Constant for all python project files

* menu.py -> Handle terminal/command-line menu

* main.py -> Main program

### CSV files
* AAPL.csv -> Apple stock data (if deleted, will grab from yahoo finance).

* DJI.csv -> Dow Jones Industrial Average stock data (if deleted, will grab from yahoo finance).

* HSI.csv -> Hang Seng Index stock data (if deleted, will grab from yahoo finance).

* GSPC.csv -> S&P 500 stock data (if deleted, will grab from yahoo finance).

## How to Run
$ python3 main.py


### Students
Informatic Engineering of State University of Surabaya (UNESA)
- Koko Himawan Permadi (19051204111) 
- Malik Dwi Yoni Fordana (17051204024)
- Roy Belmiro Virgiant (17051204016) 
