import numpy as np
import pickle

from MERA import MERA

# 1D quantum Ising model, at critical point g=1
g = 1.

s0 = np.eye(2)
sX = np.array([[0, 1], [1, 0]], dtype=float)
sZ = np.array([[1, 0], [0, -1]], dtype=float)
hloc = -np.kron(sX, sX) - g*(np.kron(sZ, s0) + np.kron(s0, sZ))/2
# hAB = hBA = hloc.reshape(2, 2, 2, 2)

# try a big hamiltonian 4*4
hbig = (0.5 * np.kron(np.eye(4), hloc) +
        np.kron(np.eye(2), np.kron(hloc, np.eye(2))) +
        0.5 * np.kron(hloc, np.eye(4))).reshape(2, 2, 2, 2, 2, 2, 2, 2)

hAB = (hbig.transpose(0, 1, 3, 2, 4, 5, 7, 6)).reshape(4, 4, 4, 4)
hBA = (hbig.transpose(1, 0, 2, 3, 5, 4, 6, 7)).reshape(4, 4, 4, 4)

model = MERA(hAB, hBA, chi=6, num_trans=1)
model.optimize(maxit=2000)
model.expand(chi=8, num_trans=1)
model.optimize(maxit=1800)
model.expand(chi=10, num_trans=2)
model.optimize(maxit=1400)
# model.expand(chi=16, num_trans=4)
# model.optimize(3000)

# save model
with open('ising.pkl', 'wb') as outp:
    pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

# # load model
# with open('ising.pkl', 'rb') as inp:
#     model = pickle.load(inp)
