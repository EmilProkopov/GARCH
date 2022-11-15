import math
from readData import readData
from ljungBox import computeLjungBox
from garch import GARCH
from distributionAssumption import DistributionAssumption
from dataSetSplitting import splitByPercentage
from modelChecking import checkModel
from forecastEvaluation import calcMeanAbsoluteError, calcRMSE
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
    calcMeanVariance

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
meanEqType = MeanEquationType.SIMPLE
distributionAssumption = DistributionAssumption.NORMAL
maxLag = 30
qStatLag = 20
forecastLength = 24
dataSetSplitCoefficient = 0.7
p = 1
q = 1
confidence = 0.95
studentDegreesOfFreedom = -1
ARMAParams = (1, 1, DistributionAssumption.NORMAL, studentDegreesOfFreedom)
ARMACoefInitGuess = [0, 0, 0]

"""
Reading data
"""
(timestamps, logReturns) = readData(filename)

"""
Plotting log returns
"""
plotReturns(timestamps, logReturns, 'Log returns')

"""
Plotting autocorrelations of the log returns series
"""
plotAutocorrelations(logReturns, maxLag, 'Autocorrelations of log returns')
plotPartialAutocorrelations(
    logReturns, maxLag, 'Partial autocorrelations of log returns')

"""
Dataset splitting
"""
((trainTimestamps, trainLogReturns), (testTimestamps, testLogReturns)) =\
    splitByPercentage(timestamps, logReturns, dataSetSplitCoefficient)

"""
ARMA mean
"""
ARMAEqModelSpecification = (getMeanEquationModel(
    distributionAssumption,
    trainTimestamps,
    ARMAParams,
    ARMACoefInitGuess))\
        if meanEqType == MeanEquationType.ARMA else None

"""
ARCH effect testing
"""
meanEquationResiduals = calcMeanEquationResiduals(
    meanEqType, trainLogReturns, None, ARMAEqModelSpecification)

meanEquationResidualsSquared = list(map(
    lambda x: x**2,
    meanEquationResiduals))

Q = computeLjungBox(meanEquationResidualsSquared, qStatLag)
print('ARCH effect test Q value:', Q)

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

paramInitGuess = [0, 0.5, 0.1]
model.fit(trainLogReturns, paramInitGuess)
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
