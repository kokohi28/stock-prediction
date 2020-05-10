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
plt.style.use('fivethirtyeight')

SOURCE = 'yahoo'
FIELD = 'Close'

SAMPLE_TRAINED = 60

# MAIN PROGRAM
if __name__ == '__main__':
  stock = ''
  dateRange = []
  percent = 0

  if CONST.DEBUG:
    stock = 'AAPL'
    dateRange = ['2012/01/01', '2019/12/17']
    percent = 80
  else:
    res = menu.menuLoop()
    # print(res)

    # Parse zone, subZone and k
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
  dateStart = dateRange[0].replace('/', '-')
  dateEnd = dateRange[1].replace('/', '-')
  
  # Fetch data from server
  print('\nFetching data from server')
  df = web.DataReader(stock, data_source=SOURCE, start=dateStart, end=dateEnd)
  
  # Also save to CSV for external use
  safeStockName = re.sub(r'\W+', '', stock)
  df.to_csv('{}.csv'.format(safeStockName))

  # Check Data Frame shape
  print('Data shape: ' + str(df.shape))

  # prepare dataset and use only Close price value
  data = df.filter([FIELD])
  dataset = data.values

  # Create len of percentage training set
  trainingDataLen = math.ceil((len(dataset) * percent) / 100)
  print('Size of trainingSet: ' + str(trainingDataLen))

  # Scale the dataset between 0 - 1
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaledData = scaler.fit_transform(dataset)
  # print(scaledData)

  # Scaled trained data
  trainData = scaledData[0:trainingDataLen , :]

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

  print('Processing the LSTM model...')
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

  print('Done Processing the LSTM model...')

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
  train = data[:trainingDataLen]
  valid = data[trainingDataLen:]
  valid['predictions'] = predictions

  # Visualize
  plt.figure(figsize=(15,8), num=safeStockName)
  plt.title('{} Prediction Price'.format(safeStockName))
  plt.xlabel('Date', fontsize=14)
  plt.ylabel('Close Price USD ($)', fontsize=14)
  plt.plot(train[FIELD])
  plt.plot(valid[[FIELD, 'predictions']])
  plt.legend(['Train', 'Actual', 'Prediction'], loc='lower right')
  plt.show()

  print('')
  print('Exiting...')
  print('')
