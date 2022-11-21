import math


def calcChangeDistance(varValuesBefore, varValuesAfter):
    return math.sqrt(sum(
        [(varValuesBefore[i] - varValuesAfter[i]) ** 2
         for i in range(len(varValuesBefore))]))


def cyclicCoordinateDescentMax(
        targetFn,
        varInitGuess,
        varBounds,
        precision=1e-12,
        precisionPerVariable=1e-12,
        maxIter=200,
        maxIterPerVariable=10):

    # dichotomy method
    def optimizeSingleVar(varVals, varInd):
        currentVarValues = varVals.copy()
        localIterCount = 0
        (minBound, maxBound) = varBounds[varInd]
        a = minBound
        b = maxBound
        c = (a + b) / 2

        def calcTargetFn(potentialVarValue):
            currentVarValues[varInd] = potentialVarValue
            return targetFn(currentVarValues)

        while (b - a >= precisionPerVariable)\
                and (localIterCount <= maxIterPerVariable):

            localIterCount += 1
            d = (a + c) / 2
            e = (c + b) / 2

            if calcTargetFn(d) >= calcTargetFn(e):
                b = c
                c = d
            else:
                a = c
                c = e

        cValue = calcTargetFn(c)

        minValue = calcTargetFn(minBound)
        if minValue >= cValue:
            return minBound

        maxValue = calcTargetFn(minBound)
        if maxValue >= cValue:
            return maxBound

        return c

    varValues = varInitGuess

    iterCount = 0
    while iterCount <= maxIter:
        iterCount += 1
        # print(iterCount)

        varValuesBefore = varValues.copy()
        for varInd in range(len(varValues)):
            newVarValue = optimizeSingleVar(varValues, varInd)
            varValues[varInd] = newVarValue

        if calcChangeDistance(varValuesBefore, varValues) <= precision:
            break

    return varValues


"""
# For testing
def __test():
    def targetFn(args):
        [x, y] = args
        return -((x - 2) ** 2 + (y - 3) ** 2)

    initGuess = [0, 0]
    varBounds = [(-100, 100), (-100, 100)]

    print(cyclicCoordinateDescentMax(targetFn, initGuess, varBounds))


__test()
"""
