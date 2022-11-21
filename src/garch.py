import math
from distributionAssumption import DistributionAssumptionUtils
from optimization import cyclicCoordinateDescentMax
from utils import splitModelParamsList
from meanEquations import\
    calcMeanEquationResiduals, calcMeanEquationPoints, calcMeanVariance


class GARCH():
    def __init__(
            self,
            p,
            q,
            distributionAssumption,
            meanEquationType,
            degreesOfFreedom=-1,
            ARMAMeanSpecification=None):

        self.p = p
        self.q = q
        self.alpha_0 = 0
        self.alpha = [0 for i in range(p + 1)]
        self.beta = [0 for i in range(q + 1)]
        self.distribution = distributionAssumption
        self.meanEquationType = meanEquationType
        self.df = degreesOfFreedom
        self.ARMAMeanSpecification = ARMAMeanSpecification

    """
    Visualisation
    """

    def getModelEquationStr(self):
        strAlpha = ' + '.join(['{}*a^2_{{t-{}}}'.format(self.alpha[i], i)
                               for i in range(1, self.p + 1)])

        strBeta = ' + '.join(
            ['{}*\\sigma^2_{{t-{}}}'.format(self.beta[j], j)
                for j in range(1, self.q + 1)])

        return '\\sigma^2_t = {} + {} + {}'.format(
            self.alpha_0, strAlpha, strBeta)

    """
    Direct parameter manipulation
    """

    def setCoefs(self, coefsList):

        if not len(coefsList) == self.p + self.q + 1:
            raise Exception('Invalid coefsList length')

        (self.alpha_0, self.alpha, self.beta) = splitModelParamsList(
            coefsList, self.p, self.q)

    """
    Model core
    """

    def computeVariance(
            self,
            aSquaredList,
            varianceList,
            t,
            alpha_0=None,
            alpha=None,
            beta=None):

        if alpha_0 is None:
            alpha_0 = self.alpha_0
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        alpha_term = sum([alpha[i] * aSquaredList[t-i]
                          for i in range(1, self.p + 1)])

        beta_term = sum([beta[j] * varianceList[t-j]
                         for j in range(1, self.q + 1)])

        return max(alpha_0 + alpha_term + beta_term, 0)

    def computeVarianceList(
            self,
            squaredResiduals,
            maxInd=None,
            initialVarAssumption=0,
            alpha_0=None,
            alpha=None,
            beta=None):

        rangeLimit = len(squaredResiduals) if (maxInd is None)\
            else min(len(squaredResiduals), maxInd + 1)

        varianceList = [initialVarAssumption for j in range(self.q)]
        for i in range(self.q, rangeLimit):
            variance_i = self.computeVariance(
                squaredResiduals,
                varianceList,
                i,
                alpha_0,
                alpha,
                beta)

            varianceList.append(variance_i)

        return varianceList

    """
    Coefficients estimation
    """

    def fit(self, logReturns, paramInitGuess):

        if not len(paramInitGuess) == self.p + self.q + 1:
            raise Exception('Invalid paramInitGuess length')

        meanResiduals = calcMeanEquationResiduals(
            self.meanEquationType,
            logReturns,
            None,
            self.ARMAMeanSpecification)

        squaredResiduals = list(map(lambda x: x**2, meanResiduals))

        meanVariance = calcMeanVariance(logReturns)

        # first param is alpha_0, then alpha_1,...,alpha_p, beta_1,...,beta_q
        def logLikelihood(params):
            (alpha_0, alpha, beta)\
                = splitModelParamsList(params, self.p, self.q)

            varianceList = self.computeVarianceList(
                squaredResiduals,
                None,
                meanVariance,
                alpha_0,
                alpha,
                beta)

            return -sum(
                [math.log(varianceList[t]) / 2 +
                 squaredResiduals[t] / (2 * varianceList[t])
                 for t in range(self.q, len(varianceList))
                 if not varianceList[t] == 0])

        alphaBounds = [(0, 1/3), *[(0, 1) for i in range(1, self.p + 1)]]
        betaBounds = [(0, 1) for i in range(0, self.q + 1)]
        varBounds = [(1e-12, 1), *alphaBounds, *betaBounds]

        paramEstimates = cyclicCoordinateDescentMax(
            logLikelihood, paramInitGuess, varBounds)

        (self.alpha_0, self.alpha, self.beta) = splitModelParamsList(
            paramEstimates, self.p, self.q)

    """
    Predictive intervals
    """

    def computePointwisePredictiveIntervals(
            self,
            logReturns,
            confidence=None):

        meanResiduals = calcMeanEquationResiduals(
            self.meanEquationType,
            logReturns,
            None,
            self.ARMAMeanSpecification)

        meanEqPoints = calcMeanEquationPoints(
            self.meanEquationType,
            logReturns,
            None,
            self.ARMAMeanSpecification)

        squaredResiduals = list(map(lambda x: x**2, meanResiduals))

        meanVariance = calcMeanVariance(logReturns)

        sdList = list(map(
            lambda x: math.sqrt(x),
            self.computeVarianceList(squaredResiduals, None, meanVariance)))

        [epsilonFrom, epsilonTo] = DistributionAssumptionUtils.\
            getConfidenceInterval(self.distribution, confidence, self.df)

        intervalMaxFrom = [0]
        intervalMaxTo = [0]

        for i in range(len(logReturns)-1):
            maxDeviation = epsilonTo * sdList[i+1]

            intervalMaxFrom.append(meanEqPoints[i] - maxDeviation)
            intervalMaxTo.append(meanEqPoints[i] + maxDeviation)

        return (intervalMaxFrom, intervalMaxTo)

    """
    Forecasting
    """

    def forecast(self, _logReturns, forecastOrigin, nPoints, confidence=0.95):
        logReturns = _logReturns.copy()[:forecastOrigin + 1]

        if forecastOrigin < max(self.p, self.q):
            raise Exception(
                "Not enough data items for forecasting from specified origin")

        meanResiduals = calcMeanEquationResiduals(
            self.meanEquationType,
            logReturns,
            None,
            self.ARMAMeanSpecification)

        meanEqPoints = calcMeanEquationPoints(
            self.meanEquationType,
            logReturns,
            None,
            self.ARMAMeanSpecification)

        meanResidualsSquared = list(map(lambda x: x**2, meanResiduals))

        meanVariance = calcMeanVariance(logReturns)

        varianceListFull = self.computeVarianceList(
            meanResidualsSquared, forecastOrigin + 1, meanVariance)

        for i in range(1, nPoints + 1):
            epsilonSquared = DistributionAssumptionUtils.\
                getRandomValue(self.distribution) ** 2

            variance_i = self.computeVariance(
                meanResidualsSquared,
                varianceListFull,
                forecastOrigin + i)

            varianceListFull.append(variance_i)
            meanResidualsSquared.append(epsilonSquared * variance_i)

        varianceForecastList = varianceListFull[forecastOrigin + 1:]

        intervalMaxFrom = []
        intervalMaxTo = []

        [_, epsilonTo] = DistributionAssumptionUtils.\
            getConfidenceInterval(self.distribution, confidence, self.df)

        accumulatedVariance = 0

        for i in range(len(varianceForecastList)):
            accumulatedVariance += varianceForecastList[i]
            maxDeviation = epsilonTo * math.sqrt(accumulatedVariance)

            intervalMaxFrom.append(meanEqPoints[i-1] - maxDeviation)
            intervalMaxTo.append(meanEqPoints[i-1] + maxDeviation)

        return (varianceForecastList, (intervalMaxFrom, intervalMaxTo))
