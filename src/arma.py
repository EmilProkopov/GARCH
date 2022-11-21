from distributionAssumption import DistributionAssumptionUtils
from optimization import cyclicCoordinateDescentMax
from utils import splitModelParamsList, composeModelParamsList


class ARMA():
    def __init__(
            self,
            p,
            q,
            distributionAssumption,
            degreesOfFreedom,
            errorStandardDeviation):

        self.p = p
        self.q = q
        self.phi_0 = 0
        self.phi = [0 for i in range(p + 1)]
        self.teta = [0 for i in range(q + 1)]
        self.distribution = distributionAssumption
        self.df = degreesOfFreedom
        self.errorStandardDeviation = errorStandardDeviation

    """
    Visualisation
    """

    def getModelEquationStr(self):
        strPhi = ' + '.join(['{}*x_{{t-{}}}'.format(self.phi[i], i)
                             for i in range(1, self.p + 1)])

        strTeta = ' + '.join(['{}*a_{{t-{}}}'.format(self.teta[j], j)
                              for j in range(1, self.q + 1)])

        return 'x_t = {} + {} + a_{{t}} - ({})'.format(
            self.phi_0, strPhi, strTeta)

    """
    Direct parameter manipulation
    """

    def setCoefs(self, coefsList):

        if not len(coefsList) == self.p + self.q + 1:
            raise Exception('Invalid coefsList length')

        (self.phi_0, self.phi, self.teta) = splitModelParamsList(
            coefsList, self.p, self.q)

    def getCoefs(self):
        return composeModelParamsList(self.phi_0, self.phi, self.teta)

    """
    Model core
    """

    def computeValue(
            self,
            xList,
            noiseList,
            t,
            phi_0=None,
            phi=None,
            teta=None):

        if phi_0 is None:
            phi_0 = self.phi_0
        if phi is None:
            phi = self.phi
        if teta is None:
            teta = self.teta

        phi_term = sum([phi[i] * xList[t-i] for i in range(1, self.p + 1)])

        teta_term = sum([teta[j] * noiseList[t-j]
                         for j in range(1, self.q + 1)])

        return phi_0 + phi_term + noiseList[t] - teta_term

    def computeValueList(
            self,
            actualList,
            phi_0=None,
            phi=None,
            teta=None,
            noiseList=None):

        sampleStartLength = max(self.p, self.q)
        nPoints = len(actualList)

        if nPoints < sampleStartLength:
            raise Exception('Samle length too small')

        if phi_0 is None:
            phi_0 = self.phi_0
        if phi is None:
            phi = self.phi
        if teta is None:
            teta = self.teta

        if noiseList is None:
            noiseList = [DistributionAssumptionUtils.getRandomValue(
                self.distribution, self.df, self.errorStandardDeviation)
                for i in range(nPoints)]

        valueList = [*actualList[:sampleStartLength]]
        for t in range(sampleStartLength, nPoints):
            valueList.append(self.computeValue(
                actualList,
                noiseList,
                t,
                phi_0,
                phi,
                teta))

        return valueList

    """
    Coefficients estimation
    """

    def fit(self, logReturns, paramInitGuess):
        if not len(paramInitGuess) == self.p + self.q + 1:
            raise Exception('Invalid paramInitGuess length')

        startSampleSize = max(self.p, self.q)
        estimationSampleSize = len(logReturns) - startSampleSize

        noiseList = [DistributionAssumptionUtils.getRandomValue(
                self.distribution, self.df, self.errorStandardDeviation)
                for i in range(len(logReturns))]

        def calcSquaredError(logReturn, modelValue):
            try:
                return (logReturn - modelValue) ** 2 / estimationSampleSize
            except:
                return float('inf') / estimationSampleSize

        # first param is alpha_0, then alpha_1,...,alpha_p, beta_1,...,beta_q
        def mse(params):
            (phi_0, phi, teta)\
                = splitModelParamsList(params, self.p, self.q)

            modelValuesList = self.computeValueList(
                logReturns,
                phi_0,
                phi,
                teta,
                noiseList)

            errors = [calcSquaredError(logReturns[i], modelValuesList[i])
                      for i in range(startSampleSize, len(logReturns))]

            return -sum(errors)

        phi0Bounds = (-10, +10)
        phiBounds = [(-10, 10) for i in range(0, self.p + 1)]
        tetaBounds = [(-10, 10) for i in range(0, self.q + 1)]
        varBounds = [phi0Bounds, *phiBounds, *tetaBounds]

        paramEstimates = cyclicCoordinateDescentMax(
            mse, paramInitGuess, varBounds)

        (self.phi_0, self.phi, self.teta) = splitModelParamsList(
            paramEstimates, self.p, self.q)

    """
    Forecasting
    """

    def forecast(self, _logReturns, forecastOrigin, nPoints):
        if forecastOrigin < max(self.p, self.q):
            raise Exception(
                "Not enough data items for forecasting from specified origin")

        valueListFull = _logReturns.copy()[:forecastOrigin + 1]

        noiseListKnown = [DistributionAssumptionUtils.getRandomValue(
            self.distribution, self.df, self.errorStandardDeviation)
            for i in range(len(valueListFull))]

        noiseListForecast = [0 for i in range(nPoints)]

        noiseListFull = [*noiseListKnown, *noiseListForecast]

        for t in range(nPoints):
            valueListFull.append(self.computeValue(
                valueListFull, noiseListFull, t))

        forecastList = valueListFull[forecastOrigin+1:]

        return forecastList
