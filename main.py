# Common
import const as CONST
import menu
import math
import re
from os import path
from datetime import datetime

# Data Fetcher
import pandas_datareader as web
import pandas as pd

# Numerical
import numpy as np

# Scaler
from sklearn.preprocessing import MinMaxScaler

# Model, LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Graph plot
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Stock server source
SOURCE = 'yahoo'

# Field used
FIELD_CLOSE = 'Close'
FIELD_DATE = 'Date'

# Wide of trained sample
SAMPLE_TRAINED = 60

# 30 Days
NEXT_PREDICTION = 30 * 24 * 60 * 60 * 1000

# Get dataframe from CSV if exist, from server when not available
def getDataFrame(stock, dateRange):
  # Format date
  dateStart = datetime.strptime(dateRange[0], '%Y/%m/%d')
  dateEnd = datetime.strptime(dateRange[1], '%Y/%m/%d')
  dateStartStr = dateRange[0].replace('/', '-')
  dateEndStr = dateRange[1].replace('/', '-')

  # Remove any non alpha-numeric char
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
    df = pd.read_csv('{}.csv'.format(safeStockName), header='infer', parse_dates=[FIELD_DATE], date_parser=dateParse)
    df.sort_values(by=FIELD_DATE)
  
    # Get minimum timestamp of CSV data
    dtMin = df.loc[df.index.min(), FIELD_DATE]
    dateMin = int(dtMin.date().strftime('%s'))
    dateMin = dateMin * 1000

    # Get maximum timestamp of CSV data
    dtMax = df.loc[df.index.max(), FIELD_DATE]
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
    mask = (df[FIELD_DATE] > dateStartStr) & (df[FIELD_DATE] <= dateEndStr)

    # Return mask and origin
    return (df.loc[mask], df)


# MAIN PROGRAM
if __name__ == '__main__':
  stock = ''
  dateRange = []
  percent = 0

  # CONST.DEBUG = True
  if CONST.DEBUG:
    stock = '^GSPC'
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

  # Stock name, remove any non alpha-numeric char
  safeStockName = re.sub(r'\W+', '', stock)

  # Check Data Frame shape
  print('Data shape: ' + str(df.shape))

  # prepare dataset and use only Close price value
  dataset = df.filter([FIELD_CLOSE]).values

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
  train = df.loc[:trainingDataLen, [FIELD_DATE, FIELD_CLOSE] ]
  valid = df.loc[trainingDataLen:, [FIELD_DATE, FIELD_CLOSE] ]
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

  # Button Predict event
  def next(event):
    # Export global var
    global dfOrigin
    global endTs
    global plt
    global ax
    global scaler
    global model

    # Add next data
    endTs = endTs + NEXT_PREDICTION
    dateNextEnd = datetime.fromtimestamp(endTs / 1000)
    print("\nNext Data until: " + dateNextEnd.strftime("%Y-%m-%d"))

    # Create mask/filter
    mask = (dfOrigin[FIELD_DATE] > dateStartStr) & (dfOrigin[FIELD_DATE] <= dateNextEnd.strftime("%Y-%m-%d"))
    dfNew = dfOrigin.loc[mask]

    # Prediction for new data 
    trainNew = dfNew.loc[:trainingDataLen, [FIELD_DATE, FIELD_CLOSE] ]
    validNew = dfNew.loc[trainingDataLen:, [FIELD_DATE, FIELD_CLOSE] ]

    # prepare dataset and use only Close price value
    datasetNew = dfNew.filter([FIELD_CLOSE]).values

    # Prepare testing dataset
    scaledNewData = scaler.fit_transform(datasetNew)
    testDataNew = scaledNewData[trainingDataLen - SAMPLE_TRAINED: , :]

    # Create dataset test x and y
    xTestNew = []
    yTestNew = datasetNew[trainingDataLen: , :]
    for i in range(SAMPLE_TRAINED, len(testDataNew)):
      xTestNew.append(testDataNew[i - SAMPLE_TRAINED:i, 0])

    # Convert test set as numpy array
    xTestNew = np.array(xTestNew)

    # Reshape test set as 3 dimension array
    xTestNew = np.reshape(xTestNew, (xTestNew.shape[0], xTestNew.shape[1], 1))

    # Models predict price values
    predictionsNew = model.predict(xTestNew)
    predictionsNew = scaler.inverse_transform(predictionsNew)

    # Get root mean square (RMSE)
    rmseNew = np.sqrt(np.mean(predictionsNew - yTestNew) ** 2)
    print('Root mean square (RMSE) - New:' + str(rmseNew))

    # Create dataframe prediction
    dfPredictionNew = pd.DataFrame(predictionsNew, columns = ['predictions'])

    # Reset the index  
    validNew = validNew.reset_index()
    dfPredictionNew = dfPredictionNew.reset_index()

    # Merge valid data and prediction data
    validNew = pd.concat([validNew, dfPredictionNew], axis=1)

    # Clear graph
    ax.clear()

    # Re-plot, add graph info
    ax.set_title('With RMSE: ' + str(rmseNew))
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Close Price USD ($)', fontsize=14)
    ax.grid(linestyle='-', linewidth='0.5', color='gray')

    # plot trained data
    ax.plot(trainNew[FIELD_DATE], trainNew[FIELD_CLOSE])

    # plot actual and predictions
    ax.plot(validNew[FIELD_DATE], validNew[[FIELD_CLOSE, 'predictions']])

    # add legend
    ax.legend(['Train', 'Actual', 'Prediction'], loc='lower right')

    # Plot
    ax.draw(renderer=None, inframe=False)
    plt.pause(0.0001)

    return

  # Create button Predict
  axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
  bnext = Button(axnext, 'PREDICT')
  bnext.on_clicked(next)

  # Add graph info
  ax.set_title('With RMSE: ' + str(rmse))
  ax.set_xlabel('Date', fontsize=14)
  ax.set_ylabel('Close Price USD ($)', fontsize=14)
  ax.grid(linestyle='-', linewidth='0.5', color='gray')
  
  # plot trained data
  ax.plot(train[FIELD_DATE], train[FIELD_CLOSE])

  # plot actual and predictions
  ax.plot(valid[FIELD_DATE], valid[[FIELD_CLOSE, 'predictions']])

  # add legend
  ax.legend(['Train', 'Actual', 'Prediction'], loc='lower right')

  # finally show graph
  plt.show()

  print('')
  print('Exiting...')
  print('')
