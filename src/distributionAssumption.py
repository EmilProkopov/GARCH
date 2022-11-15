from enum import Enum, auto
from random import gauss
import numpy as np
import scipy.stats as st


class DistributionAssumption(Enum):
    NORMAL = auto()
    STUDENT = auto()


class DistributionAssumptionUtils:
    @staticmethod
    def getRandomValue(dAssumption, degreesOfFreedom=-1):
        if dAssumption == DistributionAssumption.NORMAL:
            return gauss(0, 1)
        elif dAssumption == DistributionAssumption.STUDENT:
            return np.random.standard_t(degreesOfFreedom)

    @staticmethod
    def getConfidenceInterval(dAssumption, confidence=0.95, df=-1):
        if dAssumption == DistributionAssumption.NORMAL:
            return [
                st.norm.ppf((1-confidence)/2, loc=0, scale=1),
                st.norm.ppf(
                    confidence + (1-confidence)/2, loc=0, scale=1)]

        elif dAssumption == DistributionAssumption.STUDENT:
            return [
                st.t.ppf((1-confidence)/2, df, loc=0, scale=1),
                st.t.ppf(
                    confidence + (1-confidence)/2, df, loc=0, scale=1)]
