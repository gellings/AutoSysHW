import numpy as np
from copy import deepcopy

class FASTSLAM:
    def __init__(self, num_lm=3, alpha=[0.1, 0.01, 0.01, 0.1], sigR=0.1, sigB=0.05, particles=25):
        self.Y = np.empty((particles, 1), dtype=object)
        for i in range(particles):
            self.Y[i,0] = PARTICLE(num_lm=num_lm, alpha=alpha, sigR=sigR, sigB=sigB)
        self.M = particles
        self.w = np.empty((particles, 1))
        self.bestPidx = 0

    def propogate(self, u, dt):
        [p[0].propogate(u, dt) for p in self.Y]

    def update(self, z, lmIDs):
        w = np.ones((self.M,1))
        for i in range(lmIDs.shape[0]):
            ID = lmIDs[i:i+1,:]
            z1 = z[2*i:2*i+2,:]
            w *= np.array([[p[0].update(z1, ID) for p in self.Y]]).T
            w *= 1./np.sum(w)
        self.w = w
        self.bastPidx = np.argmax(self.w)

    def resample(self):
        scale = np.sum(self.w)
        Y_new = np.empty(self.Y.shape, dtype=object)

        r = scale*np.random.rand()/self.M
        c = self.w[0,0]
        i = 0
        for m in range(self.M):
            U = r + scale*m/self.M
            while U > c:
                i = i+1
                c = c + self.w[i,0]
            Y_new[m,0] = deepcopy(self.Y[i,0])
        self.Y = Y_new

    def est_state(self):
        return self.Y[self.bestPidx,0].est_state()

    def all_states(self):
        return [p[0].x[0:2,:] for p in self.Y]

    def lm_elps(self):
        return self.Y[self.bestPidx,0].lm_elps()

class PARTICLE:
    def __init__(self, num_lm=3, alpha=[0.1, 0.01, 0.01, 0.1], sigR=0.1, sigB=0.05):
        self.num_lm = num_lm
        self.x = np.zeros((3,1))
        self.alpha = alpha
        self.R = np.diag([sigR,sigB])
        self.lmX = np.empty((0,1))
        self.lmP = np.empty((0,2))
        self.lmIDs = np.empty((0,1), dtype=np.int)

        self.p0 = 1./10

    def propogate(self, u, dt):
        v = u[0,0]
        w = u[1,0]
        theta = self.x[2,0]

        vhat = v + (self.alpha[0]*np.abs(v) + self.alpha[1]*np.abs(w))*np.random.randn()
        what = w + (self.alpha[2]*np.abs(v) + self.alpha[3]*np.abs(w))*np.random.randn()

        self.x += np.array([[vhat/what*(-np.sin(theta) + np.sin(theta + what*dt))],
                                    [vhat/what*(np.cos(theta) - np.cos(theta + what*dt))],
                                    [what*dt]])

    def update(self, z, ID):

        if ID not in self.lmIDs:
            self.lmX = np.concatenate((self.lmX, self.x[0:2,:] + z[0,0]*np.array([[np.cos(z[1,0] + self.x[2,0])],[np.sin(z[1,0] + self.x[2,0])]])), axis=0)
            delta = np.array([[self.lmX[-2,0] - self.x[0,0]],[self.lmX[-1,0] - self.x[1,0]]])
            q_sqrt = np.linalg.norm(delta)

            H = (1/q_sqrt**2)*np.array([[q_sqrt*delta[0,0], q_sqrt*delta[1,0]],
                              [-delta[1,0], delta[0,0]]])
            Hinv = np.linalg.inv(H)
            self.lmP = np.concatenate((self.lmP, Hinv.dot(self.R).dot(Hinv.T)), axis=0)
            self.lmIDs = np.concatenate((self.lmIDs, ID),axis=0)

            return self.p0
        else:
            j = self.lmIDs.tolist().index(ID[0])
            delta = np.array([[self.lmX[2*j,0] - self.x[0,0]],[self.lmX[2*j+1,0] - self.x[1,0]]])
            q_sqrt = np.linalg.norm(delta)
            zhat = np.array([[q_sqrt],
                         [np.arctan2(delta[1,0],delta[0,0]) - self.x[2,0]]])
            if zhat[1,0] > np.pi:
                zhat[1,0] -= 2*np.pi
            if zhat[1,0] < -np.pi:
                zhat[1,0] += 2*np.pi
            assert zhat[1,0] < np.pi and zhat[1,0] > -np.pi

            H = (1/q_sqrt**2)*np.array([[q_sqrt*delta[0,0], q_sqrt*delta[1,0]],
                              [-delta[1,0], delta[0,0]]])
            P = self.lmP[2*j:2*j+2,:]
            S = H.dot(P).dot(H.T) + self.R
            Sinv = np.linalg.inv(S)
            K = P.dot(H.T).dot(Sinv)
            residual = z - zhat
            while residual[1,0] > np.pi:
                residual[1,0] -= 2*np.pi
            while residual[1,0] < -np.pi:
                residual[1,0] += 2*np.pi

            self.lmX[2*j:2*j+2,:] += K.dot(residual)
            self.lmP[2*j:2*j+2,:] = (np.eye(2) - K.dot(H)).dot(P)

            return np.linalg.det(2*np.pi*S)**(-0.5)*np.exp(-(residual).T.dot(Sinv).dot(residual)/2)[0,0]

    def est_state(self):
        return self.x.copy(), self.lmX.copy()

    def lm_elps(self):
        out = np.array([]).reshape(0,3)
        loc = np.array([]).reshape(0,2)
        for i in range(self.lmIDs.shape[0]):
            vals, vecs = np.linalg.eig(self.lmP[2*i:2+2*i,:])
            if np.all(np.isreal(vals)):
                out = np.concatenate((out, np.array([[2*np.sqrt(vals[0]), 2*np.sqrt(vals[1]), np.arctan2(vecs[1,0],vecs[0,0]) ]])))
                loc = np.concatenate((loc, self.lmX[2*i:2+2*i].T))
        return loc, out
