import numpy as np
from scipy.sparse.linalg import eigsh
from copy import deepcopy

from ncon import ncon
from layer import Layer


class MERA:
    '''
    scale-invariant MERA optimization
    modified symmetric binary MERA scheme
    '''

    def __init__(self, hAB, hBA, chi: int, num_trans: int = 0):
        '''
        hAB, hBA: local hamiltonian in tensor form
        chi: bond dimension of the scale invariant tensors
        num_trans: number of transitional layers
        '''
        # physical dimension
        self.chiP = hAB.shape[0]
        # max bond dimension
        self.chi = chi
        # determine number of layers
        min_num_layers = int(np.ceil(np.log(self.chi)/np.log(self.chiP)))
        if num_trans and num_trans < min_num_layers-1:
            raise ValueError('num_trans not enough')
        self.num_layers = num_trans+1 if num_trans else min_num_layers
        # specify dimensions at each layer
        chii = [min(self.chiP**(t+1), chi) for t in range(self.num_layers)]
        chio = [min(self.chiP**(t+2), chi) for t in range(self.num_layers)]

        # preallocate transitional layers
        self.layers = [Layer(chii[t], chio[t]) for t in range(self.num_layers)]

        # !!! ensure hamiltonian is negative definite
        self.hAB_original, self.hBA_original = hAB, hBA
        self.bias = eigsh(hAB.reshape(self.chiP**2, self.chiP**2),
                          k=1, which='LA', return_eigenvectors=False)[0]
        self.hAB = hAB - self.bias * \
            np.eye(self.chiP**2).reshape(self.chiP,
                                         self.chiP, self.chiP, self.chiP)
        self.hBA = hBA - self.bias * \
            np.eye(self.chiP**2).reshape(self.chiP,
                                         self.chiP, self.chiP, self.chiP)

    @property
    def si_layer(self):
        '''
        the scale-invariant layer is the last of layers
        '''
        return self.layers[-1]

    def optimize(self, maxit: int, verbose=True):
        for sweep in range(maxit):
            # prepare Hamiltonian at tau=0
            hAB, hBA = self.hAB, self.hBA
            # prepare scale-invariant DM
            self.si_layer.rAB, self.si_layer.rBA = self.si_layer.DM_fixed_point(
                self.si_layer.rAB, self.si_layer.rBA)
            # descend DM through all layers
            for t in reversed(range(self.num_layers-1)):
                self.layers[t].rAB, self.layers[t].rBA = self.layers[t].descend(
                    self.layers[t+1].rAB, self.layers[t+1].rBA)

            # update layers
            for t in range(self.num_layers-1):
                self.layers[t].update(
                    hAB, hBA, self.layers[t+1].rAB, self.layers[t+1].rBA)
                hAB, hBA = self.layers[t].ascend(hAB, hBA)
            # # calculate average h and update si_layer
            # hABbar, hBAbar = hAB, hBA
            # ABtemp, BAtemp = hAB, hBA
            # for _ in range(2):
            #     ABtemp, BAtemp = self.si_layer.ascend(ABtemp/2, BAtemp/2)
            #     hABbar += ABtemp
            #     hBAbar += BAtemp
            self.si_layer.update(
                hAB, hBA, self.si_layer.rAB, self.si_layer.rBA)

            if verbose and sweep % 10 == 0:
                self.display(sweep, maxit)

    def display(self, sweep, maxit):
        E0 = -4/np.pi
        Energy = (ncon([self.layers[0].rAB, self.hAB], [[1, 2, 3, 4], [1, 2, 3, 4]]) +
                  ncon([self.layers[0].rBA, self.hBA], [[1, 2, 3, 4], [1, 2, 3, 4]])) / 2 + self.bias
        if self.chiP == 4:
            Energy /= 2
        err = -(Energy-E0)/E0

        # evaluate sigma x
        sx = np.array([[0, 1], [1, 0]])
        s0 = np.eye(2)
        if self.chiP == 2:
            sx_big = (np.kron(s0, sx) + np.kron(sx, s0))/2
            ExpectX = ncon([(self.layers[0].rAB+self.layers[0].rBA)/2,
                        sx_big.reshape(2, 2, 2, 2)], [[1, 2, 3, 4], [1, 2, 3, 4]])
        elif self.chiP == 4:
            ExpectX = ncon([self.layers[0].rAB.reshape(2, 2, 2, 2, 2, 2, 2, 2), sx], [[4, 1, 2, 3, 5, 1, 2, 3], [4, 5]])

        print('Iteration: %d of %d, Energy: %f, err: %f, Mag: %e' %
              (sweep, maxit, np.real(Energy), np.real(err), np.real(ExpectX)))

    def expand(self, chi: int, num_trans: int):
        self.chi = chi
        self.num_layers = num_trans+1
        chii = [min(self.chiP**(t+1), chi) for t in range(self.num_layers)]
        chio = [min(self.chiP**(t+2), chi) for t in range(self.num_layers)]

        while len(self.layers) < self.num_layers:
            self.layers.append(deepcopy(self.si_layer))
        for t in range(self.num_layers):
            self.layers[t].expand(chii[t], chio[t])
