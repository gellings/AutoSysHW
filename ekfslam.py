import numpy as np

class EKFSLAM:
    def __init__(self, num_lm=3, alpha=[0.1, 0.01, 0.01, 0.1], sigR=0.1, sigB=0.05):
        self.num_lm = num_lm
        self.x = np.zeros((3 + 2*num_lm,1))
        self.P = 10e17*np.eye(3 + 2*num_lm)
        self.P[0:3,0:3] = np.zeros((3,3))
        self.alpha = alpha
        self.R = np.diag([sigR,sigB])
        self.lmIDs = np.empty((0,1), dtype=np.int)
        self.old = 0

    def propodate(self, u, dt):
        v = u[0,0]
        w = u[1,0]
        theta = self.x[2,0]
        F = np.zeros((3,3+2*self.num_lm))
        F[0:3,0:3] = np.eye(3)
        A = np.eye(3+2*self.num_lm) + F.T.dot(  np.array([[0,0,v/w*(-np.cos(theta) + np.cos(theta + w*dt))],
                                                          [0,0,v/w*(-np.sin(theta) + np.sin(theta + w*dt))],
                                                          [0,0,0]])                                         ).dot(F)
        G = np.array([[(-np.sin(theta) + np.sin(theta + w*dt))/w, v/w*((np.sin(theta) - np.sin(theta + w*dt))/w + np.cos(theta + w*dt)*dt)],
                      [(np.cos(theta) - np.cos(theta + w*dt))/w, v/w*((-np.cos(theta) + np.cos(theta + w*dt))/w + np.sin(theta + w*dt)*dt)],
                      [0,dt]])
        Qu = np.diag([(self.alpha[0]*np.abs(v) + self.alpha[1]*np.abs(w))**2, (self.alpha[2]*np.abs(v) + self.alpha[3]*np.abs(w))**2])
        self.x[0:3,:] += np.array([[v/w*(-np.sin(theta) + np.sin(theta + w*dt))],
                            [v/w*(np.cos(theta) - np.cos(theta + w*dt))],
                            [w*dt]])

        # print A.shape, self.P.shape, G.shape, Qu.shape
        self.P = A.dot(self.P).dot(A.T) + F.T.dot(G.dot(Qu).dot(G.T)).dot(F)

    def update(self, z, lmIDs):
        for i in range(lmIDs.shape[0]):
            ID = lmIDs[i:i+1,:]
            if ID not in self.lmIDs:
                j = self.lmIDs.shape[0]
                self.x[3+2*j:5+2*j,:] = self.x[0:2,:] + z[2*i,0]*np.array([[np.cos(z[2*i+1,0] + self.x[2,0])],[np.sin(z[2*i+1,0] + self.x[2,0])]])
                self.lmIDs = np.concatenate((self.lmIDs, ID),axis=0)

            j = self.lmIDs.tolist().index(ID[0])
            delta = np.array([[self.x[3+2*j,0] - self.x[0,0]],[self.x[4+2*j,0] - self.x[1,0]]])
            q_sqrt = np.linalg.norm(delta)
            zhat = np.array([[q_sqrt],
                             [np.arctan2(delta[1,0],delta[0,0]) - self.x[2,0]]])
            F = np.zeros((5,3+2*self.num_lm))
            F[0:3,0:3] = np.eye(3)
            F[3:5,3+2*j:5+2*j] = np.eye(2)
            H = (1/q_sqrt**2)*np.array([[-q_sqrt*delta[0,0], -q_sqrt*delta[1,0], 0, q_sqrt*delta[0,0], q_sqrt*delta[1,0]],
                          [delta[1,0], -delta[0,0], -q_sqrt**2, -delta[1,0], delta[0,0]]]).dot(F)
            Sinv = np.linalg.inv(H.dot(self.P).dot(H.T) + self.R)
            K = self.P.dot(H.T).dot(Sinv)
            residual = z[2*i:2*i+2,:] - zhat
            while residual[1,0] > np.pi:
                residual[1,0] -= 2*np.pi
            while residual[1,0] < -np.pi:
                residual[1,0] += 2*np.pi

            dist = residual.T.dot(Sinv).dot(residual)[0,0]
            if dist < 9:
                self.x = self.x + K.dot(residual)
                self.P = (np.eye(3 + 2*self.num_lm) - K.dot(H)).dot(self.P)
            else:
                print "gated a measurement", np.sqrt(dist)
            # self.x = self.x + K.dot(residual)
            # self.P = (np.eye(3 + 2*self.num_lm) - K.dot(H)).dot(self.P)
        # return K[:,0:1]

    def est_state(self):
        return self.x[0:3,:].copy(), self.x[3:,:].copy()

    def vars(self):
        return np.array([[np.sqrt(self.P[0,0])], [np.sqrt(self.P[1,1])], [np.sqrt(self.P[2,2])]])

    def state_elps(self):
        vals, vecs = np.linalg.eig(self.P[0:2,0:2])
        return 2*np.sqrt(vals[0]), 2*np.sqrt(vals[1]), np.arctan2(vecs[1,0],vecs[0,0])

    def lm_elps(self):
        out = np.array([]).reshape(0,3)
        loc = np.array([]).reshape(0,2)
        for i in range(self.lmIDs.shape[0]):
            vals, vecs = np.linalg.eig(self.P[3+2*i:5+2*i,3+2*i:5+2*i])
            if np.all(np.isreal(vals)):
                out = np.concatenate((out, np.array([[2*np.sqrt(vals[0]), 2*np.sqrt(vals[1]), np.arctan2(vecs[1,0],vecs[0,0]) ]])))
                loc = np.concatenate((loc, self.x[3+2*i:5+2*i].T))
        return loc, out
