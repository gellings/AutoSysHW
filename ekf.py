import numpy as np

class EKF:
    def __init__(self, x0=-5, y0=-3, th0=np.pi/2, alpha=[0.1, 0.01, 0.01, 0.1], lm=[(6,4),(-7,8),(6,-4)], sigR=0.1, sigB=0.05):
        self.x = np.array([[x0],[y0],[th0]])
        self.P = np.diag([2,2,np.pi/4])
        self.alpha = alpha
        self.lm = lm
        self.R = np.diag([sigR,sigB])

    def propodate(self, u, dt):
        v = u[0,0]
        w = u[1,0]
        theta = self.x[2,0]
        A = np.array([[1,0,v/w*(-np.cos(theta) + np.cos(theta + w*dt))],
                      [0,1,v/w*(-np.sin(theta) + np.sin(theta + w*dt))],
                      [0,0,1]])
        G = np.array([[(-np.sin(theta) + np.sin(theta + w*dt))/w, v/w*((np.sin(theta) - np.sin(theta + w*dt))/w + np.cos(theta + w*dt)*dt)],
                      [(np.cos(theta) - np.cos(theta + w*dt))/w, v/w*((-np.cos(theta) - np.cos(theta + w*dt))/w + np.sin(theta + w*dt)*dt)],
                      [0,dt]])
        Qu = np.diag([(self.alpha[0]*np.abs(v) + self.alpha[1]*np.abs(w))**2, (self.alpha[2]*np.abs(v) + self.alpha[3]*np.abs(w))**2])
        self.x += np.array([[v/w*(-np.sin(theta) + np.sin(theta + w*dt))],
                            [v/w*(np.cos(theta) - np.cos(theta + w*dt))],
                            [w*dt]])

        # print A.shape, self.P.shape, G.shape, Qu.shape
        self.P = A.dot(self.P).dot(A.T) + G.dot(Qu).dot(G.T)

    def update(self, z):
        for i in range(len(self.lm)):
            lma = np.array([self.lm[i]]).T
            q_sqrt = np.linalg.norm(self.x[0:2,:] - lma)
            zhat = np.array([[q_sqrt],
                             [np.arctan2(lma[1,0] - self.x[1,0], lma[0,0] - self.x[0,0]) - self.x[2,0]]])
            C = np.array([[-(lma[0,0] - self.x[0,0])/q_sqrt, -(lma[1,0] - self.x[1,0])/q_sqrt, 0],
                          [(lma[1,0] - self.x[1,0])/(q_sqrt**2), -(lma[0,0] - self.x[0,0])/(q_sqrt**2), -1]])
            inv = np.linalg.inv(C.dot(self.P).dot(C.T) + self.R)
            K = self.P.dot(C.T).dot(inv)
            self.x = self.x + K.dot(z[2*i:2*i+2,:] - zhat)
            self.P = (np.eye(3) - K.dot(C)).dot(self.P)
        return K[:,0:1]

    def est_state(self):
        return self.x

    def vars(self):
        return np.array([[np.sqrt(self.P[0,0])], [np.sqrt(self.P[1,1])], [np.sqrt(self.P[2,2])]])
