import math


def sliceByIndexes(timestamps, values, fromInd, toInd):
    return (timestamps[fromInd:toInd], values[fromInd:toInd])


def sliceByTimestamps(timestamps, values, timestampFrom, timestampTo):
    allIndexes = list(range(len(timestamps)))

    def timestampFilter(x):
        return (x >= timestampFrom) and (x < timestampTo)

    resTimestamps = [timestamps[i] for i in allIndexes
                     if timestampFilter(timestamps[i])]

    resValues = [values[i] for i in allIndexes
                 if timestampFilter(timestamps[i])]

    return (resTimestamps, resValues)


def splitByPercentage(timestamps, values, testSize):
    if testSize > 1 or testSize < 0:
        raise Exception("invalid test size")

    borderIndex = math.floor(len(timestamps) * testSize)
    return(
        (timestamps[:borderIndex], values[:borderIndex]),
        (timestamps[borderIndex:], values[borderIndex:]))
