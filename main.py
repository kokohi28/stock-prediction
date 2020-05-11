import const as CONST
import menu
import math
import re
import pandas_datareader as web
import numpy as np
import pandas as pd
# import sklearn
from sklearn.preprocessing import MinMaxScaler
# import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import datetime
from datetime import datetime
from os import path

SOURCE = 'yahoo'
FIELD = 'Close'

ADJUST_LOC = 3
SAMPLE_TRAINED = 60

# 60 Days
NEXT_SIZE = 60 * 24 * 60 * 60 * 1000


def getDataFrame(stock, dateRange):
  # Format date
  dateStart = datetime.strptime(dateRange[0], '%Y/%m/%d')
  dateEnd = datetime.strptime(dateRange[1], '%Y/%m/%d')
  dateStartStr = dateRange[0].replace('/', '-')
  dateEndStr = dateRange[1].replace('/', '-')

  safeStockName = re.sub(r'\W+', '', stock)

  # Prepare data frame
  df = None

  # Check existing CSV
  fileCsvExist = path.exists('{}.csv'.format(safeStockName))
  if not fileCsvExist:
    # Fetch data from server
    print('\nFetching data from server')
    df = web.DataReader(stock, data_source=SOURCE, start=dateStartStr, end=dateEndStr)

    # Save to CSV for later or external use
    df.to_csv('{}.csv'.format(safeStockName))
  
    return (df, df)
  else:
    # Read from CSV
    print('\nRead from CSV')
  
    # Read and parse as time series
    dateParse = lambda x: datetime.strptime(x, "%Y-%m-%d")
    df = pd.read_csv('{}.csv'.format(safeStockName), header='infer', parse_dates=['Date'], date_parser=dateParse)
    df.sort_values(by='Date')
  
    # Get minimum timestamp of CSV data
    dtMin = df.loc[df.index.min(), 'Date']
    dateMin = int(dtMin.date().strftime('%s'))
    dateMin = dateMin * 1000

    # Get maximum timestamp of CSV data
    dtMax = df.loc[df.index.max(), 'Date']
    dateMax = int(dtMax.date().strftime('%s'))
    dateMax = dateMax * 1000

    # Get start and end timestamp of input date by user
    startTs = int(dateStart.timestamp() * 1000)
    endTs = int(dateEnd.timestamp() * 1000)

    dataAppended = False
    if (startTs < dateMin):
      prevStartStr = dtMin.date().strftime("%Y-%m-%d")
      print('Fetch previous data, from:' + dateStartStr + ', to:' + prevStartStr)
      dfPrev = web.DataReader(stock, data_source=SOURCE, start=dateStartStr, end=prevStartStr)

      # Append data frame
      df = dfPrev.append(df, ignore_index=True)

      # Check the data frame
      print(df)
      dataAppended = True

    if (endTs > dateMax):
      nextStartStr = dtMax.date().strftime("%Y-%m-%d")
      print('Fetch next data, from:' + nextStartStr + ', to:' + dateEndStr)
      dfNext = web.DataReader(stock, data_source=SOURCE, start=nextStartStr, end=dateEndStr)

      # Append data frame
      df = df.append(dfNext, ignore_index=True)

      # Check the data frame
      print(df)
      dataAppended = True

    # if data appended, rewrite CSV
    if dataAppended:
      # Save to CSV for later or external use
      df.to_csv('{}.csv'.format(safeStockName))

    # Create mask/filter
    mask = (df['Date'] > dateStartStr) & (df['Date'] <= dateEndStr)

    # Return mask and origin
    return (df.loc[mask], df)


# MAIN PROGRAM
if __name__ == '__main__':
  stock = ''
  dateRange = []
  percent = 0

  # CONST.DEBUG = True
  if CONST.DEBUG:
    stock = 'AAPL'
    dateRange = ['2010/01/05', '2015/01/05']
    percent = 80
  else:
    res = menu.menuLoop()
    # print(res)

    # Parse stock, dateRange and percentage of train data
    stock = res[CONST.IDX_STOCK]
    dateRange = res[CONST.IDX_DATE_RANGE]
    percent = res[CONST.IDX_PERCENT_TRAINED]

  # Clear screen
  menu.clearScreen()
  menu.welcomeMessage()

  if stock == '' or len(dateRange) == 0 or percent == 0:
    print('')
    print('Exiting...')
    print('')
    exit()

  # Begin Process message
  print('')
  print('Processing stock: {}'.format(stock))
  print('Start-date: {}, end-date: {}'.format(dateRange[0], dateRange[1]))
  print('Percentage trained-data(%): {}'.format(percent))

  # Format date
  dateStart = datetime.strptime(dateRange[0], '%Y/%m/%d')
  dateEnd = datetime.strptime(dateRange[1], '%Y/%m/%d')
  dateStartStr = dateRange[0].replace('/', '-')
  dateEndStr = dateRange[1].replace('/', '-')
  startTs = int(dateStart.timestamp() * 1000)
  endTs = int(dateEnd.timestamp() * 1000)

  # Get data frame
  df, dfOrigin = getDataFrame(stock, dateRange)
  # print('\nUsing dataframe:')
  # print(df)

  # Stock name
  safeStockName = re.sub(r'\W+', '', stock)

  # Check Data Frame shape
  print('Data shape: ' + str(df.shape))

  # prepare dataset and use only Close price value
  dataset = df.filter([FIELD]).values

  # Create len of percentage training set
  trainingDataLen = math.ceil((len(dataset) * percent) / 100)
  print('Size of trainingSet: ' + str(trainingDataLen))

  # Scale the dataset between 0 - 1
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaledData = scaler.fit_transform(dataset)

  # Scaled trained data
  trainData = scaledData[:trainingDataLen , :]

  # Split into trained x and y
  xTrain = []
  yTrain = []
  for i in range(SAMPLE_TRAINED, len(trainData)):
    xTrain.append(trainData[i-SAMPLE_TRAINED:i , 0])
    yTrain.append(trainData[i , 0])

  # Convert trained x and y as numpy array
  xTrain, yTrain = np.array(xTrain), np.array(yTrain)
  print('x - y train shape: ' + str(xTrain.shape) + ' ' + str(yTrain.shape))

  # Reshape x trained data as 3 dimension array
  xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
  print('Expected x train shape: ' + str(xTrain.shape))
  print('')

  print('Processing the LSTM model...\n')

  # Build LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  # Compile model
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(xTrain, yTrain, batch_size=1, epochs=1)

  print('\nDone Processing the LSTM model...')

  # Prepare testing dataset
  testData = scaledData[trainingDataLen - SAMPLE_TRAINED: , :]

  # Create dataset test x and y
  xTest = []
  yTest = dataset[trainingDataLen: , :]
  for i in range(SAMPLE_TRAINED, len(testData)):
    xTest.append(testData[i - SAMPLE_TRAINED:i, 0])

  # Convert test set as numpy array
  xTest = np.array(xTest)

  # Reshape test set as 3 dimension array
  xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

  # Models predict price values
  predictions = model.predict(xTest)
  predictions = scaler.inverse_transform(predictions)

  # Get root mean square (RMSE)
  rmse = np.sqrt(np.mean(predictions - yTest) ** 2)
  print('\nRoot mean square (RMSE):' + str(rmse))

  # Add prediction for Plot
  train = df.loc[:trainingDataLen, ['Date', FIELD] ]
  valid = df.loc[trainingDataLen:, ['Date', FIELD] ]
  print('validLength: {}, predictionLength: {}'.format(len(valid), len(predictions)))

  # Create dataframe prediction
  dfPrediction = pd.DataFrame(predictions, columns = ['predictions'])

  # Reset the index  
  valid = valid.reset_index()
  dfPrediction = dfPrediction.reset_index()

  # Merge valid data and prediction data
  valid = pd.concat([valid, dfPrediction], axis=1)

  # Visualize
  fig, ax = plt.subplots(num='{} Prediction Price'.format(safeStockName))
  plt.subplots_adjust(bottom=0.2)

  def next(event):
    global df
    global endTs
    global plt
    global ax

    # Add next data
    endTs = endTs + NEXT_SIZE
    dateNextEnd = datetime.fromtimestamp(endTs / 1000)
    print("Next Data until: " + dateNextEnd.strftime("%Y-%m-%d"))

    # Create mask/filter
    mask = (dfOrigin['Date'] > dateStartStr) & (dfOrigin['Date'] <= dateNextEnd.strftime("%Y-%m-%d"))
    dfNew = dfOrigin.loc[mask]

    # Clear graph
    # ax.clear()

    # Re-plot
    # ax.set_title('{} Prediction Price'.format(safeStockName))
    # ax.set_xlabel('Date', fontsize=14)
    # ax.set_ylabel('Close Price USD ($)', fontsize=14)
    # ax.grid(linestyle='-', linewidth='0.5', color='gray')
    # ax.plot(dfNew['Date'], dfNew['Close'])
    # ax.draw(renderer=None, inframe=False)
    # plt.pause(0.0001)

    return

  axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
  bnext = Button(axnext, 'PREDICT')
  bnext.on_clicked(next)

  ax.set_title('With RMSE: ' + str(rmse))
  ax.set_xlabel('Date', fontsize=14)
  ax.set_ylabel('Close Price USD ($)', fontsize=14)
  ax.grid(linestyle='-', linewidth='0.5', color='gray')
  
  # plot trained data
  ax.plot(train['Date'], train[FIELD])

  # plot actual and predictions
  ax.plot(valid['Date'], valid[[FIELD, 'predictions']])

  # add legend
  ax.legend(['Train', 'Actual', 'Prediction'], loc='lower right')

  # finally show graph
  plt.show()

  print('')
  print('Exiting...')
  print('')
