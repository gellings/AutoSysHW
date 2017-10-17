import numpy as np

class PF:
    def __init__(self, alpha=[0.1, 0.01, 0.01, 0.1], lm=[(6,4),(-7,8),(6,-4)], sigR=0.1, sigB=0.05, particles=1000):
        self.chi = np.concatenate((20*np.random.rand(particles,2) - 10, 2*np.pi*np.random.rand(particles,1) - np.pi), axis=1)
        self.M = particles
        self.w = np.empty((particles, 1))
        self.alpha = alpha
        self.lm = lm
        self.R = np.diag([sigR,sigB,sigR,sigB,sigR,sigB])
        self.sigR = sigR
        self.sigB = sigB
        self.x = np.zeros((3,1))

    def propodate(self, u, dt):
        v = u[0,0]
        w = u[1,0]
        for i in range(self.M):
            theta = self.chi[i,2]

            vhat = v + (self.alpha[0]*np.abs(v) + self.alpha[1]*np.abs(w))*np.random.randn()
            what = w + (self.alpha[2]*np.abs(v) + self.alpha[3]*np.abs(w))*np.random.randn()

            self.chi[i:i+1,:] += np.array([[vhat/what*(-np.sin(theta) + np.sin(theta + what*dt)),
                                        vhat/what*(np.cos(theta) - np.cos(theta + what*dt)),
                                        what*dt]])

    def update(self, z):

        weight_total = 0
        for m in range(self.M):
            x = self.chi[m:m+1,:].T
            zhat = np.empty((0,1))
            w = 1
            for i in range(3):
                lma = np.array([self.lm[i]]).T
                q_sqrt = np.linalg.norm(x[0:2,:] - lma)
                zhatl = np.array([[q_sqrt],
                                 [np.arctan2(lma[1,0] - x[1,0], lma[0,0] - x[0,0]) - x[2,0]]])
                zhat = np.concatenate((zhat,zhatl),axis=0)
                # w *= 1/np.sqrt(2*np.pi*self.sigR**2)*np.exp(-((z[2*i,0] - zhatl[0,0])**2)/2/(self.sigR**2))
            self.w[m,0] = np.linalg.det(2*np.pi*self.R)**(-0.5)*np.exp(-(z - zhat).T.dot(np.linalg.inv(self.R)).dot(z - zhat)/2)[0,0]
            weight_total += self.w[m,0]

        self.resample(weight_total)

        self.x = np.atleast_2d(np.mean(self.chi, axis=0)).T


    def get_chi(self):
        return self.chi

    def resample(self, scale):
        chi_new = np.empty(self.chi.shape)

        r = scale*np.random.rand()/self.M
        c = self.w[0,0]
        i = 0
        for m in range(self.M):
            U = r + scale*m/self.M
            while U > c:
                i = i+1
                c = c + self.w[i,0]
            chi_new[m:m+1,:] = self.chi[i:i+1,:]
        self.chi = chi_new

    def est_state(self):
        return self.x

    def vars(self):
        return np.atleast_2d(np.var(self.chi,axis=0)**0.5).T
