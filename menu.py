import os
import const as CONST
import datetime

# Const
MENU_ROOT = 0
MENU_SPECIFY_DATE = 1
MENU_SPECIFY_PERCENT_TRAINED = 2

currMenu = MENU_ROOT

stockList = ['AAPL', '^DJI', '^HSI', '^GSPC']

def welcomeMessage():
  print('##############################################################################')
  print('####                                                                      ####')
  print('####   Stock Prediction using Long short-term memory (LSTM)               ####')
  print('####                                                                      ####')
  print('####   BY : - Malik Dwi Yoni Fordana (17051204024)                        ####')
  print('####        - Roy Belmiro Virgiant (17051204016)                          ####')
  print('####        - Koko Himawan Permadi (19051204111)                          ####')
  print('####                                                                      ####')
  print('##############################################################################')
  return

def validateDate(date_text):
  try:
    datetime.datetime.strptime(date_text, '%Y/%m/%d')
    return True
  except ValueError:
    return False

def menuSpecifyPercentTrained():
  print('\nEnter trained data percentage (%) :')
  print('')
  print('Press [B] for Back')
  return

def menuSpecifyDate():
  print('\nEnter period of stock, start date - end date :')
  print('example : 2012/01/01-2019/12/17')
  print('')
  print('Press [B] for Back')
  return

def menuRoot():
  print('\nSelect Stock:')
  print('1. Apple (AAPL)')
  print('2. Dow Jones Industrial Average (^DJI)')
  print('3. Hang Seng Index (^HSI)')
  print('4. S&P 500 (^GSPC)')
  print('')
  print('Press [Q] for Exit')
  return

def handleInputDate(inputVal):
  if inputVal == 'B' or \
     inputVal == 'b':
    return (-1, [])
  else:
    dateSplit = inputVal.split('-')
    if len(dateSplit) < 2:
      print('\nRange INVALID... (Press any key to continue)')
      input('')
      return (0, [])
    else:
      if validateDate(dateSplit[0]) == False:
        print('\nDate start INVALID... (Press any key to continue)')
        input('')
        return (0, [])
      
      if validateDate(dateSplit[1]) == False:
        print('\nDate end INVALID... (Press any key to continue)')
        input('')
        return (0, [])
      
      return (1, dateSplit)

def handleInputPercentTrained(inputVal):
  if inputVal == 'B' or \
     inputVal == 'b':
    return -1
  else:
    if inputVal.isnumeric():
      num = int(inputVal)
      if num == 0 or \
         num > 100:
        print('\nPercentage INVALID... (Press any key to continue)')
        input('')
        return 0
      else:
        return num
    else:
      print('\nPercentage INVALID... (Press any key to continue)')
      input('')
      return 0

def clearScreen():
  os.system('cls' if os.name == 'nt' else 'clear')
  return

def menuLoop():
  loopMenu = True
  global currMenu

  stock = ''
  dateRange = []
  percentTrained = 0

  while loopMenu:
    try:
      # Clear screen
      clearScreen()

      # Display Welcome
      welcomeMessage()
      
      # Display Input
      inputMsg = ''
      if currMenu == MENU_ROOT:
        menuRoot()
        inputMsg = 'Select : '
      elif currMenu == MENU_SPECIFY_DATE:
        menuSpecifyDate()
        inputMsg = 'Specify : '
      elif currMenu == MENU_SPECIFY_PERCENT_TRAINED:
        menuSpecifyPercentTrained()
        inputMsg = 'Percentage : '
      
      # Get Input
      inputVal = input(inputMsg)

      # Listen Quit Input
      if inputVal == 'Q' or \
        inputVal == 'q':
        stock = ''
        dateRange = []
        percentTrained = 0

        loopMenu = False
      else:
        # Root
        if currMenu == MENU_ROOT:
          if inputVal.isnumeric():
            num = int(inputVal)
            if num == 0 or \
              num > len(stockList):
              print('\nSelection INVALID... (Press any key to continue)')
              input('')
            else:
              stock = stockList[num - 1]
              currMenu = currMenu + 1
          else:
            print('\nSelection INVALID... (Press any key to continue)')
            input('')

        # Date
        elif currMenu == MENU_SPECIFY_DATE:
          res, dateRange = handleInputDate(inputVal)
          if res < 0:
            currMenu = currMenu - 1
          elif res == 0:
            pass
          elif res > 0:
            currMenu = currMenu + 1

        # Percent trained
        elif currMenu == MENU_SPECIFY_PERCENT_TRAINED:
          percentTrained = handleInputPercentTrained(inputVal)
          if percentTrained < 0:
            currMenu = currMenu - 1
          elif percentTrained == 0:
            pass
          elif percentTrained > 0:
            #  EXIT MENU LOOP
            loopMenu = False

    except KeyboardInterrupt:
      stock = ''
      dateRange = []
      percentTrained = 0

      loopMenu = False

  return (stock, dateRange, percentTrained)