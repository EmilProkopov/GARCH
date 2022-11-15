import csv


def readData(fileName: str):
    dates = []
    logReturns = []

    with open(fileName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            dates.append(int(row[0]))
            logReturns.append(float(row[1]))

    return (dates, logReturns)
