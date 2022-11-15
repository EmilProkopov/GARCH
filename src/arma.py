from distributionAssumption import DistributionAssumptionUtils
from optimization import cyclicCoordinateDescentMax
from utils import splitModelParamsList, composeModelParamsList


class ARMA():
    def __init__(
            self,
            p,
            q,
            distributionAssumption,
            degreesOfFreedom):

        self.p = p
        self.q = q
        self.phi_0 = 0
        self.phi = [0 for i in range(p + 1)]
        self.teta = [0 for i in range(q + 1)]
        self.distribution = distributionAssumption
        self.df = degreesOfFreedom

    """
    Visualisation
    """

    def getModelEquationStr(self):
        strPhi = ' + '.join(['{}*x_{{t-{}}}'.format(self.phi[i], i)
                             for i in range(1, self.p + 1)])

        strTeta = ' + '.join(['{}*a_{{t-{}}}'.format(self.teta[j], j)
                              for j in range(1, self.q + 1)])

        return 'x_t = {} + {} + a_{{t}} + {}'.format(
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
            nPoints,
            sampleStart,
            phi_0=None,
            phi=None,
            teta=None):

        sampleStartLength = max(self.p, self.q)

        if len(sampleStart) < sampleStartLength:
            raise Exception('Invalid sampleStart length')

        if phi_0 is None:
            phi_0 = self.phi_0
        if phi is None:
            phi = self.phi
        if teta is None:
            teta = self.teta

        noiseList = [DistributionAssumptionUtils.getRandomValue(
            self.distribution, self.df)
            for i in range(nPoints)]

        valueList = [*sampleStart[:sampleStartLength]]
        for t in range(sampleStartLength, nPoints):
            valueList.append(self.computeValue(
                valueList,
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

        # first param is alpha_0, then alpha_1,...,alpha_p, beta_1,...,beta_q
        def mse(params):
            (phi_0, phi, teta)\
                = splitModelParamsList(params, self.p, self.q)

            modelValuesList = self.computeValueList(
                len(logReturns),
                logReturns,
                phi_0,
                phi,
                teta)

            return -sum([(logReturns[i] - modelValuesList[i]) ** 2
                         for i in range(startSampleSize, len(logReturns))])\
                / startSampleSize

        phiBounds = [(-10, 10) for i in range(0, self.p + 1)]
        tetaBounds = [(-10, 10) for i in range(0, self.q + 1)]
        varBounds = [(-10, 10), *phiBounds, *tetaBounds]

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
            self.distribution, self.df)
            for i in range(len(valueListFull))]

        noiseListForecast = [0 for i in range(nPoints)]

        noiseListFull = [*noiseListKnown, *noiseListForecast]

        for t in range(nPoints):
            valueListFull.append(self.computeValue(
                valueListFull, noiseListFull, t))

        forecastList = valueListFull[forecastOrigin+1:]

        return forecastList
