from ljungBox import computeLjungBox
from functools import reduce
from plotting import QQplot
import math


def checkModel(meanEquationResiduals, standardDeviations, maxLag, dAssumption):
    if not len(meanEquationResiduals) == len(standardDeviations):
        raise Exception("list lengths must be equal")

    print('Model check summary:')

    # standardisedResiduals
    sr = [meanEquationResiduals[i] / standardDeviations[i]
          for i in range(len(meanEquationResiduals))
          if standardDeviations[i] > 0]

    # mean equation adequacy
    print('Q statistics of standardised residuals: {}'.format(
        computeLjungBox(sr, maxLag)))

    print('Q statistics of squared standardised residuals: {}'.format(
        computeLjungBox(list(map(lambda x: x**2, sr)), maxLag)))

    # skewness and kurtosis
    sampleMean = reduce(lambda accum, value: accum + value, sr) / len(sr)
    sampleVariance = reduce(
        lambda accum, value: accum + (value - sampleMean) ** 2,
        sr) / len(sr)

    sampleDeviation = math.sqrt(sampleVariance)

    skewness = reduce(
        lambda accum, value:
            accum + ((value - sampleMean) / sampleDeviation) ** 3,
        sr) / len(sr)

    kurtosis = reduce(
        lambda accum, value:
            accum + ((value - sampleMean) / sampleDeviation) ** 4,
        sr) / len(sr)

    print('skewness: {}'.format(skewness))
    print('kurtosis: {}'.format(kurtosis))

    # quantile-to-quantile plot
    QQplot(sr, dAssumption)
