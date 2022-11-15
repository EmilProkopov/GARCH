import csv
import math
import datetime

monthDict = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

lastMonthDayDict = {
    'Jan': '31',
    'Feb': '28',
    'Mar': '31',
    'Apr': '30',
    'May': '31',
    'Jun': '30',
    'Jul': '31',
    'Aug': '31',
    'Sep': '30',
    'Oct': '31',
    'Nov': '30',
    'Dec': '31'
}


def parseDateNoDay(dateStr):
    splitted = dateStr.split(' ')
    month = monthDict[splitted[0]]
    day = lastMonthDayDict[splitted[0]]
    year = '20' + splitted[1]
    return int(datetime.datetime(int(year), int(month), int(day)).timestamp())


def parseDateWithDay(dateStr):
    splitted = dateStr.replace(',', '').split(' ')
    month = monthDict[splitted[0]]
    day = splitted[1]
    year = splitted[2]
    return int(datetime.datetime(int(year), int(month), int(day)).timestamp())


def computeLogReturnsTestData():
    returnsPlusOne = []
    dates = []
    with open('../data/test-data.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')

        rowCounter = -1
        for row in reader:
            rowCounter += 1
            if (rowCounter == 0):  # skip title
                continue

            nonEmptyIndex = 1
            while nonEmptyIndex < len(row) and row[nonEmptyIndex] == '':
                nonEmptyIndex += 1

            returnsPlusOne.append(float(row[nonEmptyIndex]) + 1)
            dates.append(int(row[0]))

    logReturns = list(map(math.log, returnsPlusOne))

    with open('../data/test-data-log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(0, len(dates)):
            writer.writerow([str(dates[i]), str(logReturns[i])])


def computeLogReturnsWIG(fileName, outputFileName):
    dates = []
    prices = []
    with open('../data/{}'.format(fileName), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        rowCounter = -1
        for row in reader:
            rowCounter += 1
            if (rowCounter == 0):  # skip title
                continue

            priceOpen = float(row[1])
            priceClose = float(row[4])
            prices.append((priceOpen + priceClose) / 2)
            dates.append(int(
                datetime.datetime.strptime(row[0], '%Y-%m-%d').timestamp()))

    dates = dates[1:]
    logReturns = []
    for i in range(0, len(prices) - 1):
        logReturns.append(math.log(prices[i+1] / prices[i]))

    with open('../data/{}'.format(outputFileName), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(0, len(dates)):
            writer.writerow([str(dates[i]), str(logReturns[i])])


def computeLogReturnsEthers(fileName, outputFileName, parseDateFn):
    dates = []
    prices = []
    with open('../data/{}'.format(fileName), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        rowCounter = -1
        for row in reader:
            rowCounter += 1
            if (rowCounter == 0):  # skip title
                continue

            prices.append(float(row[1].replace(",", "")))
            dates.append(parseDateFn(row[0]))

    prices.reverse()
    dates.reverse()

    dates = dates[1:]
    logReturns = []
    for i in range(0, len(prices) - 1):
        logReturns.append(math.log(prices[i+1] / prices[i]))

    with open('../data/{}'.format(outputFileName), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(0, len(dates)):
            writer.writerow([str(dates[i]), str(logReturns[i])])


computeLogReturnsTestData()
computeLogReturnsWIG('wig20 Daily.csv', 'wig20 Daily-log.csv')
computeLogReturnsWIG('wig20 Monthly.csv', 'wig20 Monthly-log.csv')
computeLogReturnsWIG('wig20 Weekly.csv', 'wig20 Weekly-log.csv')
computeLogReturnsEthers('Ethereum Historical Data Monthly.csv',
                        'Ethereum Historical Data Monthly-log.csv',
                        parseDateNoDay)
computeLogReturnsEthers('Ethereum Historical Data Weekly.csv',
                        'Ethereum Historical Data Weekly-log.csv',
                        parseDateWithDay)
computeLogReturnsEthers('Ethereum Historical Data Daily.csv',
                        'Ethereum Historical Data Daily-log.csv',
                        parseDateWithDay)
