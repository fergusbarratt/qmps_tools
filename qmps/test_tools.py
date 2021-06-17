from xmps.iMPS import iMPS
from numpy.random import randn
from numpy import allclose
import numpy as np
import qmps.tools as tools

N = 3
xs = [randn(8, 8) + 1j * randn(8, 8) for _ in range(N)]
As = [iMPS().random(2, 2).mixed() for _ in range(N)]


def test_tensor_to_unitary():
    for AL, AR, C in As:
        U, passed = tools.tensor_to_unitary(AL[0], testing=True)
        assert passed
        AL_new = tools.unitary_to_tensor(U)
        assert allclose(AL_new, AL[0])


def test_unitary_to_tensor():
    N = 10
    As = []
    for i in range(N):
        U = np.linalg.qr(np.random.randn(16, 16))[0]
        As.append(tools.unitary_to_tensor(U))
    
    for A in As:
        assert allclose(A.shape, (2, 8, 8))
