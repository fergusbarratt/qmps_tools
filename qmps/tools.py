from numpy import eye, concatenate, allclose, swapaxes, tensordot
from numpy import array, pi as π, arcsin, sqrt, real, imag, split
from numpy import zeros, block, diag, log2
from numpy.random import rand, randint, randn
from numpy.linalg import svd, qr
from scipy.linalg import null_space


def random_unitary(*args):
    return qr(randn(*args))[0]


def svals(A):
    return svd(A)[1]


def from_real_vector(v):
    """helper function - put list of elements (real, imaginary) into a complex vector"""
    re, im = split(v, 2)
    return re + im * 1j


def to_real_vector(A):
    """takes a matrix, breaks it down into a real vector"""
    re, im = real(A).reshape(-1), imag(A).reshape(-1)
    return concatenate([re, im], axis=0)


def eye_like(A):
    """eye_like: identity same shape as A"""
    return eye(A.shape[0])


def cT(tensor):
    """H: Hermitian conjugate of last two indices of a tensor

    :param tensor: tensor to conjugate
    """
    return swapaxes(tensor.conj(), -1, -2)


def direct_sum(A, B):
    """direct sum of two matrices"""
    (a1, a2), (b1, b2) = A.shape, B.shape
    O = zeros((a2, b1))
    return block([[A, O], [O.T, B]])


def unitary_extension(Q, D=None):
    """extend an isometry to a unitary (doesn't check its an isometry)"""
    s = Q.shape
    flipped = False
    N1 = null_space(Q)
    N2 = null_space(Q.conj().T)

    if s[0] > s[1]:
        Q_ = concatenate([Q, N2], 1)
    elif s[0] < s[1]:
        Q_ = concatenate([Q.conj().T, N1], 1).conj().T
    else:
        Q_ = Q

    if D is not None:
        if D > Q_.shape[0]:
            Q_ = direct_sum(Q_, eye(D - Q_.shape[0]))

    return Q_


def environment_to_unitary(v):
    """put matrix in form
    ↑ ↑
    | |
    ___
     v
    ___
    | |
    """
    v = v.reshape(1, -1) / norm(v)
    vs = null_space(v).conj().T
    return concatenate([v, vs], 0).T


def environment_from_unitary(u):
    """matrix out of form
    ↑ ↑
    | |
    ___
     v
    ___
    | |
    """
    return (u @ array([1, 0, 0, 0])).reshape(2, 2)


def tensor_to_unitary(A, testing=False):
    """given a left isometric tensor A, put into a unitary.
    NOTE: A should be left canonical: No checks!
    """
    d, D, _ = A.shape
    iso = A.transpose([1, 0, 2]).reshape(D * d, D)
    U = unitary_extension(iso)
    if testing:
        passed = (
            allclose(cT(iso) @ iso, eye(2))
            and allclose(U @ cT(U), eye(4))
            and allclose(cT(U) @ U, eye(4))
            and allclose(U[: iso.shape[0], : iso.shape[1]], iso)
            and allclose(
                tensordot(U.reshape(2, 2, 2, 2), array([1, 0]), [2, 0]).reshape(4, 2),
                iso,
            )
        )
        return U, passed

    #  ↑ j
    #  | |
    #  ---
    #   u  = i--A--j
    #  ---      |
    #  | |      σ
    #  i σ

    return U


def unitary_to_tensor(U):
    n = int(log2(U.shape[0]))
    return (
        tensordot(U.reshape(*2 * n * [2]), array([1, 0]), [n, 0])
        .reshape(2 ** (n - 1), 2, 2 ** (n - 1))
        .transpose([1, 0, 2])
    )
