def computeAutocorrelations(data, maxLag):
    maxLag += 1
    acorrs = maxLag * [0]
    acorrs[0] = 1
    mean = sum(data) / len(data)
    var = sum([(x - mean) ** 2 for x in data]) / len(data)
    normalizedData = [x - mean for x in data]

    for lag in range(1, maxLag):
        tmp = [normalizedData[lag:][i] * normalizedData[:-lag][i]
               for i in range(len(data) - lag)]
        c = sum(tmp) / len(data) / var
        acorrs[lag] = c

    return acorrs


# Durbin-Levinson Algorithm
def computePartialAutocorrelations(data, maxLag):
    acorrs = computeAutocorrelations(data, maxLag)
    maxLag += 1
    pacorrs = maxLag * [0]
    pacorrs[0] = 1

    pacfVals = [[None for j in range(0, maxLag)] for i in range(0, maxLag)]
    pacfVals[1][1] = acorrs[1]

    def pacf(n, k):
        if pacfVals[n][k] is not None:
            pass

        elif n == k:
            numerator = acorrs[n] - sum([acorrs[n-k] * pacf(n-1, k)
                                         for k in range(1, n-1)])

            denominator = 1 - sum([acorrs[k] * pacf(n-1, k)
                                   for k in range(1, n-1)])

            pacfVals[n][k] = numerator / denominator

        else:
            pacfVals[n][k] = pacf(n-1, k) - pacf(n, n) * pacf(n-1, n-k)

        return pacfVals[n][k]

    for lag in range(1, maxLag):
        pacorrs[lag] = pacf(lag, lag)

    return pacorrs
