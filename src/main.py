import math
from readData import readData
from ljungBox import computeLjungBox
from garch import GARCH
from distributionAssumption import DistributionAssumption
from dataSetSplitting import splitByPercentage
from modelChecking import checkModel
from forecastEvaluation import calcMeanAbsoluteError, calcRMSE
from autocorrelations import computePartialAutocorrelations
from plotting import\
    plotReturns,\
    plotAutocorrelations,\
    plotPartialAutocorrelations,\
    plotModel,\
    plotModelForecast
from meanEquations import\
    MeanEquationType,\
    calcMeanEquationResiduals,\
    getMeanEquationModel,\
    calcMeanEquationPointsForecast,\
    calcMeanVariance,\
    calcMeanEquationPoints,\
    getARMAMeanEqForFittedModel

# filename = '../data/test-data-log.csv'
# filename = '../data/wig20 Monthly-log.csv'
# filename = '../data/wig20 Weekly-log.csv'
# filename = '../data/wig20 Daily-log.csv'
filename = '../data/Ethereum Historical Data Monthly-log.csv'
# filename = '../data/Ethereum Historical Data Weekly-log.csv'
# filename = '../data/Ethereum Historical Data Daily-log.csv'

"""
Constants
"""
meanEqType = MeanEquationType.ARMA
distributionAssumption = DistributionAssumption.NORMAL
maxLag = 30
qStatLag = 20
forecastLength = 24
dataSetSplitCoefficient = 0.7
confidence = 0.95
studentDegreesOfFreedom = -1
p = 2
q = 1
GARCHCoefInitGuess = [0, 0.5, 0.5, 0.1]
ARMAp = 2
ARMAq = 1
ARMACoefInitGuess = [0, *[0 for i in range(ARMAp)], *[0 for i in range(ARMAq)]]
ARMAErrorStandardDeviation = 0.25

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


print('ARMA mean equation full data: {}'.format(
        getARMAMeanEqForFittedModel(ARMAEqModelSpecificationFullData)))

meanEquationPointsFullData = calcMeanEquationPoints(
    meanEqType, logReturns, None, ARMAEqModelSpecificationFullData)

logReturnsWithMeanEq = [logReturns[i] - meanEquationPointsFullData[i]
                        for i in range(len(logReturns))]

plotAutocorrelations(
    logReturnsWithMeanEq,
    maxLag,
    'Log returns autocorrelations with ARMA mean equation')

plotPartialAutocorrelations(
    logReturnsWithMeanEq,
    maxLag,
    'Log returns partial autocorrelations with ARMA mean equation')

partialAutocorrelations = computePartialAutocorrelations(
    logReturnsWithMeanEq, maxLag)[1:]

absPartialAutocorrelations = list(map(
    lambda x: abs(x),
    partialAutocorrelations))

print('Maximim absolute partial autocorrelation value: {}'.format(
    max(absPartialAutocorrelations)))


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
    meanEqType, trainLogReturns, None, ARMAEqModelSpecification)

meanEquationResidualsSquared = list(map(
    lambda x: x**2,
    meanEquationResiduals))

Q_ARCH = computeLjungBox(meanEquationResidualsSquared, qStatLag)
print('ARCH effect test Q value with ARCH mean:', Q_ARCH)

"""
Model declaration and coefficients estimation
"""
model = GARCH(
    p,
    q,
    distributionAssumption,
    meanEqType,
    studentDegreesOfFreedom,
    ARMAEqModelSpecification)

model.fit(trainLogReturns, GARCHCoefInitGuess)
# model.setCoefs([0.00092, 0.186, 0.853])
# model.setCoefs(0.00092, [0, 0.186], [0, 0.853])
print('Model equation: {}'.format(model.getModelEquationStr()))

"""
Model checking
"""
meanTrainVariance = calcMeanVariance(trainLogReturns)

trainVarianceList = model.computeVarianceList(
    trainLogReturns,
    len(trainLogReturns) - 1,
    meanTrainVariance)

checkModel(
    trainLogReturns,
    list(map(math.sqrt, trainVarianceList)),
    maxLag,
    distributionAssumption)

"""
Model visualisation
"""
trainMeanEqPrecomputedSimple = getMeanEquationModel(
    MeanEquationType.SIMPLE, trainLogReturns)

trainMeanEqResiduals = calcMeanEquationResiduals(
    meanEqType, trainLogReturns, None, ARMAEqModelSpecification)

trainMeanEqResidualsSquared = list(map(
    lambda x: x**2,
    trainMeanEqResiduals))

trainModelPointwiseVolatility = model.computeVarianceList(
    trainMeanEqResidualsSquared, None, meanTrainVariance)

trainModelPointwisePredictionIntervals = model.\
    computePointwisePredictiveIntervals(trainLogReturns, confidence)

plotModel(
    trainTimestamps,
    trainLogReturns,
    meanEqType,
    trainMeanEqPrecomputedSimple,
    ARMAEqModelSpecification,
    trainModelPointwiseVolatility,
    trainModelPointwisePredictionIntervals)

"""
Forecasting
"""
(forecastVol, forecastPredictiveIntervals) = model.forecast(
    trainLogReturns,
    len(trainLogReturns) - 1,
    forecastLength)

meanEqForecast = calcMeanEquationPointsForecast(
    meanEqType, trainLogReturns, forecastLength, ARMAEqModelSpecification)

plotModelForecast(
    testTimestamps[:forecastLength],
    testLogReturns[:forecastLength],
    forecastVol,
    meanEqType,
    trainMeanEqPrecomputedSimple,
    ARMAEqModelSpecification,
    meanEqForecast,
    forecastPredictiveIntervals)

testMeanEqResiduals = calcMeanEquationResiduals(
    meanEqType, testLogReturns, None, ARMAEqModelSpecification)

testMeanEqResidualsSquared = list(map(
    lambda x: x**2,
    testMeanEqResiduals))

print('\nForecast evaluation:')
print('Mean absolute error: {}'.format(calcMeanAbsoluteError(
    testMeanEqResidualsSquared[:forecastLength], forecastVol)))

print('Root mean squared error: {}'.format(calcRMSE(
    testMeanEqResidualsSquared[:forecastLength], forecastVol)))
