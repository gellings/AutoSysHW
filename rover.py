import numpy as np

class Rover:
    def __init__(self, x0=-5, y0=-3, th0=np.pi/2, alpha=[0.1, 0.01, 0.01, 0.1], lm=[(6,4),(-7,8),(6,-4)], sigR=0.1, sigB=0.05):
        self.x = x0
        self.y = y0
        self.theta = th0
        self.alpha = alpha
        self.lm = lm
        self.sigR = sigR
        self.sigB = sigB

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
        for i in range(len(self.lm)):
            lma = np.array([self.lm[i]]).T
            rng = np.linalg.norm([[self.x],[self.y]] - lma) + self.sigR*np.random.randn()
            brng = np.arctan2(lma[1,0] - self.y, lma[0,0] - self.x) - self.theta + self.sigB*np.random.randn()
            mes = np.concatenate((mes, np.array([[rng],[brng]])))

        return mes

    def true_state(self):
        return np.array([[self.x], [self.y], [self.theta]])
