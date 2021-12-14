import numpy as np
from numpy import linalg as LA
from scipy.stats import unitary_group
from ncon import ncon


class Layer:
    '''
    one translation invariant layer in MERA, with modified binary scheme
    '''

    def __init__(self, chii: int, chio: int = 0, u=None, w=None):
        '''
        the tau-th layer
        '''
        # bond dimensions (in from finer lattice, out to coarse lattice)
        self.chii = chii
        self.chio = chio if chio else chii
        # initialize tensors
        self.u = u if u else self.random_disentangler(self.chii)
        self.w = w if w else self.random_isometry(self.chii, self.chio)
        # initialize density matrix
        # self.rAB = self.rBA = np.eye(
        #     chii**2).reshape(chii, chii, chii, chii)/chii**2
        self.rAB = np.random.rand(chii, chii, chii, chii)
        self.rBA = np.random.rand(chii, chii, chii, chii)

    def update(self, hAB, hBA, rAB, rBA):
        '''
        h from layer tau-1
        rho from layer tau+1
        '''
        self.u = self.svd_update(self.u_env(hAB, hBA, rAB, rBA), 2)
        self.w = self.svd_update(self.w_env(hAB, hBA, rAB, rBA), 2)

    def export(self):
        data = {}
        data['u'] = self.u
        data['w'] = self.w
        return data

    def u_env(self, hAB, hBA, rAB, rBA):
        '''
        three terms, two symmetric
        '''
        # last two w's are v and vdagger, the refl of w
        left = ncon([hAB, rBA, self.w, self.w.conj(), self.u.conj(), self.w, self.w.conj()], [
                    [7, 8, 10, -1], [4, 3, 9, 2], [10, -3, 9], [7, 5, 4], [8, -2, 5, 6], [1, -4, 2], [1, 6, 3]])
        # right = left.transpose(1, 0, 3, 2)
        mid = ncon([hBA, rBA, self.w, self.w.conj(), self.u.conj(), self.w, self.w.conj()], [
                   [7, 8, -1, -2], [3, 6, 2, 5], [1, -3, 2], [1, 9, 3], [7, 8, 9, 10], [4, -4, 5], [4, 10, 6]])

        env = left+mid/2
        env = env+env.transpose(1, 0, 3, 2)
        return env

    def w_env(self, hAB, hBA, rAB, rBA):
        '''
        four terms
        '''
        left = ncon([hAB, rBA, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                    [7, 8, -1, 9], [4, 3, -3, 2], [7, 5, 4], [9, 10, -2, 11], [8, 10, 5, 6], [1, 11, 2], [1, 6, 3]])
        mid = ncon([hBA, rBA, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                   [1, 2, 3, 4], [10, 7, -3, 6], [-1, 11, 10], [3, 4, -2, 8], [1, 2, 11, 9], [5, 8, 6], [5, 9, 7]])
        right = ncon([hAB, rBA, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                     [5, 7, 3, 1], [10, 9, -3, 8], [-1, 11, 10], [4, 3, -2, 2], [4, 5, 11, 6], [1, 2, 8], [7, 6, 9]])

        env_AB = ncon([hBA, rAB, self.w, self.w.conj(), self.w.conj()], [
                      [3, 7, 2, -1], [5, 6, 4, -3], [2, 1, 4], [3, 1, 5], [7, -2, 6]])

        return left+mid+right+env_AB

    @staticmethod
    def svd_update(env: np.ndarray, leftnum: int):
        '''
        optimize a tensor using its linearized environment
        leftnum: how many vertices on one side of the output tensor
        '''
        shape = env.shape
        env_mat = env.reshape(
            np.prod(shape[0:leftnum]), np.prod(shape[leftnum:len(shape)]))
        U, _, V = LA.svd(env_mat, full_matrices=False)
        return (-U@V).reshape(shape)

    @staticmethod
    def random_disentangler(chi):
        half = unitary_group.rvs(chi)
        return np.kron(half, half).reshape(chi, chi, chi, chi).transpose(1, 0, 2, 3)

    @staticmethod
    def random_isometry(chii, chio):
        w = unitary_group.rvs(chii**2)
        return w[:, :chio].reshape(chii, chii, chio)

    def ascend(self, hAB, hBA):
        '''
        ascend local hamiltonian: tau-1 -> tau
        '''
        hBAout = ncon([hAB, self.w, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                      [6, 4, 1, 2], [1, 3, -3], [6, 7, -1], [2, 5, 3, 9], [4, 5, 7, 10], [8, 9, -4], [8, 10, -2]])
        hBAout += hBAout.transpose(1, 0, 3, 2)
        hBAout += ncon([hBA, self.w, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                       [3, 4, 1, 2], [5, 6, -3], [5, 7, -1], [1, 2, 6, 9], [3, 4, 7, 10], [8, 9, -4], [8, 10, -2]])

        hABout = ncon([hBA, self.w, self.w.conj(), self.w, self.w.conj()], [
                      [3, 6, 2, 5], [2, 1, -3], [3, 1, -1], [5, 4, -4], [6, 4, -2]])

        return hABout, hBAout

    def descend(self, rAB, rBA):
        '''
        descend density operator: tau+1 -> tau
        '''
        rABout = ncon([rBA, self.w, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                      [9, 3, 4, 2], [-3, 5, 4], [-1, 10, 9], [-4, 7, 5, 6], [-2, 7, 10, 8], [1, 6, 2], [1, 8, 3]])
        rABout = (rABout+rABout.transpose(1, 0, 3, 2))/2

        rBAout = ncon([rBA, self.w, self.w.conj(), self.u, self.u.conj(), self.w, self.w.conj()], [
                      [3, 6, 2, 5], [1, 7, 2], [1, 9, 3], [-3, -4, 7, 8], [-1, -2, 9, 10], [4, 8, 5], [4, 10, 6]])
        rBAout = (rBAout+rBAout.transpose(1, 0, 3, 2))/2

        return rABout, rBAout

    def expand(self, chii: int, chio: int = 0):
        chio = chio if chio else chii
        self.u = np.pad(self.u, (0, chii-self.chii))
        self.w = np.pad(self.w, ((0, chii-self.chii),
                        (0, chii-self.chii), (0, chio-self.chio)))

        self.rAB = np.pad(self.rAB, (0, chii-self.chii))
        self.rBA = np.pad(self.rBA, (0, chii-self.chii))

        self.chii = chii
        self.chio = chio

    def DM_fixed_point(self, rAB, rBA, power=5):
        '''
        obtain scale-invariant density matrix
        power method
        '''
        for _ in range(power):
            rAB, rBA = self.descend(rAB, rBA)
            # normalize
            rAB /= ncon([rAB], [[1, 2, 1, 2]])
            rBA /= ncon([rBA], [[1, 2, 1, 2]])
            # symmetrize
            rAB = (rAB + rAB.conj().transpose(2, 3, 0, 1))/2
            rAB = (rAB + rAB.transpose(1, 0, 3, 2))/2
            rBA = (rBA + rBA.conj().transpose(2, 3, 0, 1))/2
            rBA = (rBA + rBA.transpose(1, 0, 3, 2))/2
        return rAB, rBA

    def scaling_dim(self):
        if self.chii != self.chio:
            raise RuntimeError('Conformal functions are only for the scale-invariant layer')
        chi = self.chii