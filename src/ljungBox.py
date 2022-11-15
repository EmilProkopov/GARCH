from autocorrelations import computeAutocorrelations


def computeLjungBox(sample, maxLag):
    n = len(sample)
    acorrs = computeAutocorrelations(sample, maxLag)
    return n * (n + 2) * \
        sum([acorrs[k] ** 2 / (n - k) for k in range(1, maxLag + 1)])
