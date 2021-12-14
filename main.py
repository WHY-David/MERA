import numpy as np
import pickle

from MERA import MERA

# 1D quantum Ising model, at critical point g=1
g = 1.

s0 = np.eye(2)
sX = np.array([[0, 1], [1, 0]], dtype=float)
sZ = np.array([[1, 0], [0, -1]], dtype=float)
hloc = -np.kron(sX, sX) - g*(np.kron(sZ, s0) + np.kron(s0, sZ))/2
hAB = hBA = hloc.reshape(2, 2, 2, 2)

model = MERA(hAB, hBA, chi=6)
model.optimize(maxit=2000)
model.expand(chi=8, num_trans=3)
model.optimize(maxit=2000)
model.expand(chi=10, num_trans=3)
model.optimize(maxit=2000)
# model.expand(chi=16, num_trans=4)
# model.optimize(3000)

# save model
with open('ising.pkl', 'wb') as outp:
    pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

# # load model
# with open('ising.pkl', 'rb') as inp:
#     model = pickle.load(inp)
