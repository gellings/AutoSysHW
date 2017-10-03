import numpy as np

class UKF:
    def __init__(self, x0=-5, y0=-3, th0=np.pi/2, alpha=[0.1, 0.01, 0.01, 0.1], lm=[(6,4),(-7,8),(6,-4)], sigR=0.1, sigB=0.05, al=1.0 ,kap=1.0):
        self.x = np.array([[x0],[y0],[th0]])
        self.P = np.diag([2,2,np.pi/4])
        self.alpha = alpha
        self.lm = lm
        self.R = np.diag([sigR,sigB])

        self.lamb = al**2*(7 + kap) - 7
        self.gamma = np.sqrt(7 + self.lamb)
        self.w_m = np.concatenate(([[self.lamb/(7 + self.lamb)]], np.ones((1,2*7))/2/(7 + self.lamb)), axis=1)
        self.w_c = np.concatenate(([[self.lamb/(7 + self.lamb) + (1 - al**2 + 2)]], np.ones((1,2*7))/2/(7 + self.lamb)), axis=1)


    def propodate(self, u, dt):
        v = u[0,0]
        w = u[1,0]
        theta = self.x[2,0]

        x_a = np.concatenate((self.x, np.zeros((4, 1))), axis=0)
        Qu = np.diag([(self.alpha[0]*np.abs(v) + self.alpha[1]*np.abs(w))**2, (self.alpha[2]*np.abs(v) + self.alpha[3]*np.abs(w))**2])
        Z_2 = np.zeros((2,2))
        Z_3 = np.zeros((3,2))
        P_a = np.asarray(np.bmat([[self.P, Z_3, Z_3], [Z_3.T, Qu, Z_2], [Z_3.T, Z_2.T, self.R]]))

        L = np.linalg.cholesky(P_a)
        chi_a = np.concatenate((x_a, x_a + self.gamma*L, x_a - self.gamma*L), axis=1)

        # g(u + chi_u,chi_x)
        for i in range(2*7 + 1):
            v = u[0,0] + chi_a[3,i]
            w = u[1,0] + chi_a[4,i]
            theta = chi_a[2,i]
            chi_a[0:3,i:i+1] += np.array([[v/w*(-np.sin(theta) + np.sin(theta + w*dt))],
                                [v/w*(np.cos(theta) - np.cos(theta + w*dt))],
                                [w*dt]])
        self.x = np.atleast_2d(np.sum(self.w_m*chi_a[0:3,:],axis=1)).T

        self.P = (self.w_c*(chi_a[0:3,:] - self.x)).dot((chi_a[0:3,:] - self.x).T)

        self.chi_a = chi_a

    def update(self, z, lm_idx):
        i = lm_idx

        lma = np.array([self.lm[i]]).T
        Zbar = np.empty((2,2*7 + 1))
        for j in range(2*7 + 1):
            q_sqrt = np.linalg.norm(self.chi_a[0:2,j:j+1] - lma)
            Zbar[:,j:j+1] = np.array([[q_sqrt],
                         [np.arctan2(lma[1,0] - self.chi_a[1,j], lma[0,0] - self.chi_a[0,j]) - self.chi_a[2,j]]]) + self.chi_a[5:7,j:j+1]
        zhat = np.atleast_2d(np.sum(self.w_m*Zbar,axis=1)).T

        S = (self.w_c*(Zbar - zhat)).dot((Zbar - zhat).T)
        P_Ct = (self.w_c*(self.chi_a[0:3,:] - self.x)).dot((Zbar - zhat).T)
        K = P_Ct.dot(np.linalg.inv(S))

        self.x = self.x + K.dot(z[2*i:2*i+2,:] - zhat)
        self.P = self.P - K.dot(S).dot(K.T)

        return K[:,0:1]

    def est_state(self):
        return self.x

    def vars(self):
        return np.array([[np.sqrt(self.P[0,0])], [np.sqrt(self.P[1,1])], [np.sqrt(self.P[2,2])]])
