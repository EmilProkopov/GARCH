import matplotlib.pyplot as plt
import numpy as np
from autocorrelations import\
    computeAutocorrelations, computePartialAutocorrelations
from distributionAssumption import DistributionAssumption
from meanEquations import calcMeanEquationResiduals, calcMeanEquationPoints


def plotReturns(timestamps, logReturns, label):
    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        logReturns,
        label=label,
        color='k',
        linewidth=0.9)

    plt.tick_params(axis='x', labelrotation=90)
    plt.margins(x=0)
    plt.xlabel('Time')
    plt.ylabel('Log returns')
    plt.legend()
    plt.show()


def plotAutocorrelations(values, maxLag, label):
    acorrs = computeAutocorrelations(values, maxLag)

    plt.bar(range(0, maxLag + 1), acorrs, label=label, width=0.2, color='k')
    plt.xlabel('Lag')
    plt.grid()
    plt.legend()
    plt.show()


def plotPartialAutocorrelations(values, maxLag, label):
    pacorrs = computePartialAutocorrelations(values, maxLag)

    plt.bar(range(0, maxLag + 1), pacorrs, label=label, width=0.2, color='k')
    plt.xlabel('Lag')
    plt.grid()
    plt.legend()
    plt.show()


def QQplot(_values, assumption, degreesOfFreedom=-1, numberOfQuantiles=30):
    values = _values.copy()
    theoretical = []
    xLabel = ''
    yLabel = 'Empirical values'

    if assumption == DistributionAssumption.NORMAL:
        theoretical = np.random.normal(size=len(values) * 100)
        xLabel = 'Normal distribution'

    elif assumption == DistributionAssumption.STUDENT:
        theoretical = np.random.standard_t(
            degreesOfFreedom, size=len(values) * 100)

        xLabel = 'Stident t distribution with {} degrees offreedom'.format(
            degreesOfFreedom)

    values = np.sort(values)
    theoretical = np.sort(theoretical)

    theoreticalQuantiles = np.quantile(
        theoretical, np.linspace(0, 1, numberOfQuantiles))

    empiricalQuantiles = np.quantile(
        values, np.linspace(0, 1, numberOfQuantiles))

    minDiagonal = np.min([np.min(theoretical), np.min(values)])
    maxDiagonal = np.max([np.min(theoretical), np.max(values)])

    plt.plot(
        [minDiagonal, maxDiagonal],
        [minDiagonal, maxDiagonal],
        color='r',
        marker=None,
        linewidth=0.9)

    plt.scatter(
        theoreticalQuantiles, empiricalQuantiles, color='k', marker='.')

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()


def plotModel(
        timestamps,
        logReturns,
        meanEqType,
        meanEqPrecomputed,
        ARMAPrecomputed,
        modelPointwiseVolatility,
        modelPointwisePredictiveIntervals):

    meanEqResiduals = calcMeanEquationResiduals(
        meanEqType, logReturns, meanEqPrecomputed, ARMAPrecomputed)

    meanEqPoints = calcMeanEquationPoints(
        meanEqType, logReturns, meanEqPrecomputed, ARMAPrecomputed)

    meanEqResidualsSquared = list(map(lambda x: x**2, meanEqResiduals))

    (predMaxFrom, predMaxTo) = modelPointwisePredictiveIntervals

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqResiduals,
        label='Mean equation residuals',
        color='k',
        linewidth=0.9,
        alpha=0.3)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqPoints,
        label='Mean equation',
        color='g',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqResidualsSquared,
        label='Actual volatility',
        color='b',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        modelPointwiseVolatility,
        label='Model estimated volatility',
        color='r',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        predMaxFrom,
        label='Pointwise prediction interval upper bound',
        color='salmon',
        linewidth=0.9,
        linestyle='--')

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        predMaxTo,
        color='salmon',
        linewidth=0.9,
        linestyle='--')

    plt.xlabel('Time')
    plt.tick_params(axis='x', labelrotation=90)
    plt.margins(x=0)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()


def plotModelForecast(
        timestamps,
        actualLogReturns,
        forecastVol,
        meanEqType,
        meanEqPrecomputed,
        ARMAPrecmputed,
        meanEqPointsForecast,
        modelForecastPointwisePredictiveIntervals):

    meanEqResiduals = calcMeanEquationResiduals(
        meanEqType, actualLogReturns, meanEqPrecomputed, ARMAPrecmputed)

    meanEqPoints = calcMeanEquationPoints(
        meanEqType, actualLogReturns, meanEqPrecomputed, ARMAPrecmputed)

    meanEqResidualsSquared = list(map(lambda x: x**2, meanEqResiduals))

    (predMaxFrom, predMaxTo) = modelForecastPointwisePredictiveIntervals

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqResiduals,
        label='Actual mean equation residuals',
        color='k',
        linewidth=0.9,
        alpha=0.3)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqPoints,
        label='Actual mean equation',
        color='g',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqPointsForecast,
        label='Mean equation forecast',
        color='lime',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        meanEqResidualsSquared,
        label='Actual volatility',
        color='b',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        forecastVol,
        label='Model volatility forecast',
        color='r',
        linewidth=0.9)

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        predMaxFrom,
        label='Forecast interval upper bound',
        color='salmon',
        linewidth=0.9,
        linestyle='--')

    plt.plot(
        np.asarray(timestamps, dtype='datetime64[s]'),
        predMaxTo,
        color='salmon',
        linewidth=0.9,
        linestyle='--')

    plt.xlabel('Time')
    plt.tick_params(axis='x', labelrotation=90)
    plt.margins(x=0)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
