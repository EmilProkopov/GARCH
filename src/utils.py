def splitModelParamsList(paramsList, p, q):
    # returns (alpha_0, [alpha_1,...,alpha_p], [beta_1,...,beta_q])
    return (
        paramsList[0],
        [0, *paramsList[1:p + 1]],
        [0, *paramsList[p + 1:]])


def composeModelParamsList(alpha_0, alpha, beta):
    return [alpha_0, *alpha[1:], *beta[1:]]
