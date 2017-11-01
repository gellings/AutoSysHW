import numpy as np

class Rover:
    def __init__(self, x0, alpha=[0.1, 0.01, 0.01, 0.1], lm=[(6,4),(-7,8),(6,-4)], sigR=0.1, sigB=0.05, maxrange=10, view=45):
        self.x = x0[0,0]
        self.y = x0[1,0]
        self.theta = x0[2,0]
        self.alpha = alpha
        self.lm = lm
        self.sigR = sigR
        self.sigB = sigB
        self.maxrange = maxrange
        self.view = view*np.pi/180

    def propogate_dynamics(self, u, dt):
        v = u[0,0]
        w = u[1,0]
        vhat = v + (self.alpha[0]*np.abs(v) + self.alpha[1]*np.abs(w))*np.random.randn()
        what = w + (self.alpha[2]*np.abs(v) + self.alpha[3]*np.abs(w))*np.random.randn()
        self.x += vhat/what*(-np.sin(self.theta) + np.sin(self.theta + what*dt))
        self.y += vhat/what*(np.cos(self.theta) - np.cos(self.theta + what*dt))
        self.theta += what*dt

    def get_mesurement(self):
        mes = np.array([]).reshape(0,1)
        IDs = np.array([], dtype=np.int).reshape(0,1)
        for i in range(len(self.lm)):
            lma = np.array([self.lm[i]]).T
            rng = np.linalg.norm([[self.x],[self.y]] - lma) + self.sigR*np.random.randn()
            brng = np.arctan2(lma[1,0] - self.y, lma[0,0] - self.x) - self.theta + self.sigB*np.random.randn()
            if brng > np.pi:
                brng -= 2*np.pi
            if brng < -np.pi:
                brng += 2*np.pi
            if rng < self.maxrange and brng < self.view/2 and brng > -self.view/2:
                IDs = np.concatenate((IDs, np.array([[i]])))
                mes = np.concatenate((mes, np.array([[rng],[brng]])))

        return mes, IDs

    def true_state(self):
        return np.array([[self.x], [self.y], [self.theta]])
