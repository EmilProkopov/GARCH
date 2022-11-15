import math


def calcMeanAbsoluteError(observedVolatility, forecastVolatility):
    length = len(observedVolatility)
    if not length == len(forecastVolatility):
        raise Exception('Lists of different lengths')

    return sum([abs(observedVolatility[i] - forecastVolatility[i])
                for i in range(0, length)]) / length


def calcRMSE(observedVolatility, forecastVolatility):
    length = len(observedVolatility)
    if not length == len(forecastVolatility):
        raise Exception('Lists of different lengths')

    return math.sqrt(sum(
        [(observedVolatility[i] - forecastVolatility[i]) ** 2
         for i in range(0, length)]) / length)
