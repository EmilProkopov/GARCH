from enum import Enum, auto
from arma import ARMA
from distributionAssumption import DistributionAssumption


class MeanEquationType(Enum):
    SIMPLE = auto()
    ARMA = auto()


defaultARMAParams = (1, 1, DistributionAssumption.NORMAL, 2, 10)


def getMeanEquationModel(
        assumption,
        sample,
        ARMAParams=defaultARMAParams,
        ARMACoefInitGuess=None):

    if assumption == MeanEquationType.SIMPLE:
        mean = sum(sample) / len(sample)
        return mean

    elif assumption == MeanEquationType.ARMA:
        (p, q, distributionAssumption, degreesOfFreedom, errorSD) = ARMAParams
        if ARMACoefInitGuess is None:
            ARMACoefInitGuess = [
                0, *[0 for i in range(p)], *[0 for i in range(q)]]

        model = ARMA(p, q, distributionAssumption, degreesOfFreedom, errorSD)
        model.fit(sample, ARMACoefInitGuess)

        return (ARMAParams, model.getCoefs())


def getARMAMeanEqForFittedModel(ARMAEqModel):
    (ARMAParams, ARMACoefs) = ARMAEqModel
    (p, q, distributionAssumption, degreesOfFreedom, errorSD) = ARMAParams
    model = ARMA(p, q, distributionAssumption, degreesOfFreedom, errorSD)
    model.setCoefs(ARMACoefs)

    return model.getModelEquationStr()


def calcMeanEquationPoints(
        assumption,
        sample,
        precomputedMean=None,
        ARMAEqModel=None):

    if assumption == MeanEquationType.SIMPLE:
        if precomputedMean is None:
            mean = sum(sample) / len(sample)
        else:
            mean = precomputedMean

        return [mean for i in range(len(sample))]

    elif assumption == MeanEquationType.ARMA:
        if ARMAEqModel is None:
            ARMAEqModel = getMeanEquationModel(assumption, sample)

        (ARMAParams, ARMACoefs) = ARMAEqModel
        (p, q, distributionAssumption, degreesOfFreedom, errorSD) = ARMAParams
        model = ARMA(p, q, distributionAssumption, degreesOfFreedom, errorSD)
        model.setCoefs(ARMACoefs)

        return model.computeValueList(sample)


def calcMeanEquationResiduals(
        assumption,
        sample,
        precomputedMean=None,
        ARMAEqModel=None):

    meanPoints = calcMeanEquationPoints(
        assumption, sample, precomputedMean, ARMAEqModel)

    return list(map(lambda x, y: x - y, sample, meanPoints))


def calcMeanEquationPointsForecast(
        assumption,
        sample,
        nPoints,
        ARMAEqModel=None):

    if assumption == MeanEquationType.SIMPLE:
        mean = sum(sample) / len(sample)
        return [mean for i in range(nPoints)]

    elif assumption == MeanEquationType.ARMA:
        if ARMAEqModel is None:
            ARMAEqModel = getMeanEquationModel(assumption, sample)

        (ARMAParams, ARMACoefs) = ARMAEqModel
        (p, q, distributionAssumption, degreesOfFreedom, errorSD) = ARMAParams
        model = ARMA(p, q, distributionAssumption, degreesOfFreedom, errorSD)
        model.setCoefs(ARMACoefs)

        return model.forecast(sample, len(sample) - 1, nPoints)


def calcMeanVariance(sample):
    mean = sum(sample) / len(sample)
    variance = sum([(x - mean) ** 2 for x in sample]) / len(sample)
    return variance
