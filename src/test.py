from readData import readData
from ljungBox import computeLjungBox
from distributionAssumption import DistributionAssumption
from dataSetSplitting import splitByPercentage
from autocorrelations import computePartialAutocorrelations
from plotting import\
    plotReturns,\
    plotAutocorrelations,\
    plotPartialAutocorrelations,\
    plotMean
from meanEquations import\
    MeanEquationType,\
    calcMeanEquationResiduals,\
    getMeanEquationModel,\
    calcMeanEquationPoints,\
    getARMAMeanEqForFittedModel

# filename = '../data/test-data-log.csv'
filename = '../data/wig20 Monthly-log.csv'
# filename = '../data/wig20 Weekly-log.csv'
# filename = '../data/wig20 Daily-log.csv'
# filename = '../data/Ethereum Historical Data Monthly-log.csv'
# filename = '../data/Ethereum Historical Data Weekly-log.csv'
# filename = '../data/Ethereum Historical Data Daily-log.csv'

"""
Constants
"""
meanEqType = MeanEquationType.ARMA
distributionAssumption = DistributionAssumption.NORMAL
maxLag = 50
qStatLag = 20
forecastLength = 24
dataSetSplitCoefficient = 0.7
confidence = 0.95
studentDegreesOfFreedom = -1
p = 1
q = 1
GARCHCoefInitGuess = [0, 0.9, 0.1]
ARMAp = 1
ARMAq = 0
ARMACoefInitGuess = [0, *[0 for i in range(ARMAp)], *[0 for i in range(ARMAq)]]
ARMAErrorStandardDeviation = 0.01

ARMAParams = (
    ARMAp,
    ARMAq,
    DistributionAssumption.NORMAL,
    studentDegreesOfFreedom,
    ARMAErrorStandardDeviation)

"""
Reading data
"""
(timestamps, logReturns) = readData(filename)

"""
Dataset splitting
"""
((trainTimestamps, trainLogReturns), (testTimestamps, testLogReturns)) =\
    splitByPercentage(timestamps, logReturns, dataSetSplitCoefficient)

"""
ARMA mean
"""
"""
if meanEqType == MeanEquationType.ARMA:
    ARMAEqModelSpecification = getMeanEquationModel(
        meanEqType,
        trainLogReturns,
        ARMAParams,
        ARMACoefInitGuess)

    print('ARMA mean equation: {}'.format(
        getARMAMeanEqForFittedModel(ARMAEqModelSpecification)))
else:
    ARMAEqModelSpecification = None
"""

"""
Plotting log returns
"""
plotReturns(timestamps, logReturns, 'Log returns')


"""
Plotting autocorrelations of the log returns series
"""
plotAutocorrelations(logReturns, maxLag, 'Log returns autocorrelations')

plotPartialAutocorrelations(
    logReturns, maxLag, 'Log returns partial autocorrelations')


ARMAEqModelSpecificationFullData = getMeanEquationModel(
        meanEqType,
        logReturns,
        ARMAParams,
        ARMACoefInitGuess)

meanEquationPointsFullData = calcMeanEquationPoints(
    meanEqType, logReturns, None, ARMAEqModelSpecificationFullData)


if meanEqType == MeanEquationType.ARMA:
    print('ARMA mean equation full data: {}'.format(
        getARMAMeanEqForFittedModel(ARMAEqModelSpecificationFullData)))

logReturnsWithMeanEq = [logReturns[i] - meanEquationPointsFullData[i]
                        for i in range(len(logReturns))]

plotAutocorrelations(
    logReturnsWithMeanEq,
    maxLag,
    'Mean eq residuals ACF WIG20 daily')

plotPartialAutocorrelations(
    logReturnsWithMeanEq,
    maxLag,
    'Mean eq residuals PACF WIG20 daily')

partialAutocorrelations = computePartialAutocorrelations(
    logReturnsWithMeanEq, maxLag)[1:]

absPartialAutocorrelations = list(map(
    lambda x: abs(x),
    partialAutocorrelations))

print('Maximim absolute partial autocorrelation value: {}'.format(
    max(absPartialAutocorrelations)))

plotMean(timestamps,
         logReturns,
         meanEquationPointsFullData,
         'Mean equation WIG20 daily')

"""
ARCH effect testing
"""
meanEquationResidualsSimple = calcMeanEquationResiduals(
    MeanEquationType.SIMPLE, trainLogReturns, None, None)

meanEquationResidualsSimpleSquared = list(map(
    lambda x: x**2,
    meanEquationResidualsSimple))

Q_simple = computeLjungBox(meanEquationResidualsSimpleSquared, qStatLag)
print('ARCH effect test Q value without ARCH:', Q_simple)

meanEquationResiduals = calcMeanEquationResiduals(
    meanEqType, trainLogReturns, None, ARMAEqModelSpecificationFullData)

meanEquationResidualsSquared = list(map(
    lambda x: x**2,
    meanEquationResiduals))

Q_ARCH = computeLjungBox(meanEquationResidualsSquared, qStatLag)
print('ARCH effect test Q value with ARCH mean:', Q_ARCH)
