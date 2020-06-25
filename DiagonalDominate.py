import numpy as np


#  Produces array of size A with True on dominant terms
def DiagonalAbsSum(A):
    dim = A.shape[0]
    A_abs = abs(A)
    #  Subtracts abs sum of row except that element from abs of that element
    mapping = 2*A_abs - np.transpose(np.ones((dim, dim)) *
                                     np.sum(A_abs, axis=1))
    mapping = (mapping > 0)
    return mapping


#  Determines whether or not a convergent matrix is possible
def ConvergencePotential(A):
    dim = A.shape[0]
    mapping = DiagonalAbsSum(A)
    Trueness = 0
    for i in range(dim):
        if 1 in mapping[:, i]:
            Trueness += 1
    for i in range(dim):
        if 1 in np.transpose(mapping)[:, i]:
            Trueness += 1
    if Trueness == 2 * dim:
        return True, mapping
    else:
        return False, A


#  Separates the array into rows
def ArrayRowSeparate(A):
    x = []
    for i in range(A.shape[0]):
        x.append(A[i, :])
    return x


#  Checks if a mapping is diagonally dominant
def DiagTrue(A):
    if np.trace(A) == A.shape[0]:
        return True
    else:
        return False


#  Shuffles multiple arrays in the same way to preserve equations
def shuffle_in_unison_scary(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    return a, b, c


#  Using the mapping shuffles until array is diagonally dominant
def SimpleDiagonalDominator(A, B):
    Possible, mapping = ConvergencePotential(A)
    A = ArrayRowSeparate(A)
    mapping = ArrayRowSeparate(mapping)
    B = ArrayRowSeparate(B)
    if Possible is True:
        while DiagTrue(np.array(mapping)) is False:
            mapping, A, B = shuffle_in_unison_scary(mapping, A, B)
        return np.asarray(A), np.asarray(B)
    else:
        return np.asarray(A), np.asarray(B)
