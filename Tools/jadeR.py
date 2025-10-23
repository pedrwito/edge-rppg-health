"""
JADE algorithm for Independent Component Analysis (ICA)

This module implements the JADE algorithm for blind source separation.
Based on the original MATLAB implementation by Jean-Francois Cardoso.

References:
    Cardoso, J. (1999) High-order contrasts for independent component analysis. 
    Neural Computation, 11(1): 157-192.
"""

from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float64, matrix, multiply, ndarray, \
    sign, sin, sqrt, zeros
from numpy.linalg import eig, pinv


def jadeR(X, m=None, verbose=False):
    """
    Blind separation of real signals with JADE.

    This function implements JADE, an Independent Component Analysis (ICA)
    algorithm developed by Jean-Francois Cardoso.

    Parameters:
    -----------
    X : numpy.ndarray
        An n x T data matrix (n sensors, T samples). Must be a NumPy array
        or matrix.
    m : int, optional
        Number of independent components to extract. Output matrix B will
        have size m x n so that only m sources are extracted. Defaults to None,
        in which case m == n.
    verbose : bool, optional
        Print info on progress. Default is False.

    Returns:
    --------
    numpy.ndarray
        An m*n matrix B, such that Y = B * X are separated sources extracted 
        from the n * T data matrix X.

    Raises:
    -------
    AssertionError
        If input validation fails.
    """
    # Input validation
    assert isinstance(X, ndarray), \
           "X (input data matrix) is of the wrong type (%s)" % type(X)
    origtype = X.dtype  # remember to return matrix B of the same type
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert isinstance(verbose, bool), \
           "verbose parameter should be either True or False"

    # n is number of input signals, T is number of samples
    [n, T] = X.shape
    assert n < T, "number of sensors must be smaller than number of samples"

    # Number of sources defaults to number of sensors
    if m is None:
        m = n
    assert m <= n, \
        "number of sources (%d) is larger than number of sensors (%d )" % (m, n)

    if verbose:
        print("jade -> Looking for %d sources" % m)
        print("jade -> Removing the mean value")
    X -= X.mean(1)

    # whitening & projection onto signal subspace
    # ===========================================

    if verbose: 
        print("jade -> Whitening the data")
    # An eigen basis for the sample covariance matrix
    [D, U] = eig((X * X.T) / float(T))
    # Sort by increasing variances
    k = D.argsort()
    Ds = D[k]
    # The m most significant princip. comp. by decreasing variance
    PCs = arange(n-1, n-m-1, -1)

    # --- PCA  ----------------------------------------------------------
    # At this stage, B does the PCA on m components
    B = U[:, k[PCs]].T

    # --- Scaling  ------------------------------------------------------
    # The scales of the principal components
    scales = sqrt(Ds[PCs])
    # Now, B does PCA followed by a rescaling = sphering
    B = diag(1./scales) * B
    # --- Sphering ------------------------------------------------------
    X = B * X

    # We have done the easy part: B is a whitening matrix and X is white.

    del U, D, Ds, k, PCs, scales

    # Estimation of the cumulant matrices
    # ===================================

    if verbose: 
        print("jade -> Estimating cumulant matrices")

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T
    # Dim. of the space of real symm matrices
    dimsymm = (m * (m + 1)) // 2
    # number of cumulant matrices
    nbcm = dimsymm
    # Storage for cumulant matrices
    CM = matrix(zeros([m, m*nbcm], dtype=float64))
    R = matrix(eye(m, dtype=float64))
    # Temp for a cum. matrix
    Qij = matrix(zeros([m, m], dtype=float64))
    # Temp
    Xim = zeros(m, dtype=float64)
    # Temp
    Xijm = zeros(m, dtype=float64)

    # will index the columns of CM where to store the cum. mats.
    Range = arange(m)

    for im in range(m):
        Xim = X[:, im]
        Xijm = multiply(Xim, Xim)
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * (R[:, im] * R[:, im].T)
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:, jm])
            Qij = sqrt(2) * (multiply(Xijm, X).T * X / float(T)
                            - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m

    # Joint diagonalization of the cumulant matrices
    # ==============================================

    V = matrix(eye(m, dtype=float64))

    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:, Range])
        On = On + (Diag*Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM, CM).sum(axis=0)).sum(axis=1) - On
    # A statistically scaled threshold on `small" angles
    seuil = 1.0e-6 / sqrt(T)
    # sweep number
    encore = True
    sweep = 0
    # Total number of rotations
    updates = 0
    # Number of rotations in a given seep
    upds = 0
    g = zeros([2, nbcm], dtype=float64)
    gg = zeros([2, 2], dtype=float64)
    G = zeros([2, 2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper
    # ============================
    if verbose: 
        print("jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose: 
            print("jade -> Sweep #%3d" % sweep)
        sweep = sweep + 1
        upds = 0
        for p in range(m-1):
            for q in range(p+1, m):

                Ip = arange(p, m*nbcm, m)
                Iq = arange(q, m*nbcm, m)

                # computation of Givens angle
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s], [s, c]])
                    pair = array([p, q])
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, concatenate([Ip, Iq])] = \
                      append(c*CM[:, Ip]+s*CM[:, Iq], -s*CM[:, Ip]+c*CM[:, Iq],
                             axis=1)
                    On = On + Gain
                    Off = Off - Gain

        if verbose: 
            print("completed in %d rotations" % upds)
        updates = updates + upds

    if verbose: 
        print("jade -> Total of %d Givens rotations" % updates)

    # A separating matrix
    # ===================
    B = V.T * B

    # Permute the rows of the separating matrix B to get the most energetic
    # components first. Here the **signals** are normalized to unit variance.
    # Therefore, the sort is according to the norm of the columns of
    # A = pinv(B)

    if verbose: 
        print("jade -> Sorting the components")

    A = pinv(B)

    keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
    B = B[keys, :]
    # % Is this smart ?
    B = B[::-1, :]

    if verbose: 
        print("jade -> Fixing the signs")
    b = B[:, 0]
    # just a trick to deal with sign == 0
    signs = array(sign(sign(b)+0.1).T)[0]
    B = diag(signs) * B

    return B.astype(origtype)
