import numpy as np

class UUV:
    def __init__(self):
        self.m = 100 #kg
        self.b = 20 #N-s/m
        self.v = 0
        self.x = 0
        self.pos_noise_var = 0.0001 #m**2
        self.vel_noise_var = 0.01 #m**2/s**2
        self.mes_noise_var = 0.001 #m**2

    def propogate_dynamics(self, F, dt):
        N = 10
        for _ in range(N):
            self.v += (dt/N)*((F - self.b*self.v)/self.m + np.sqrt(self.vel_noise_var)/N*np.random.randn())
            self.x += (dt/N)*(self.v + np.sqrt(self.pos_noise_var)/N*np.random.randn())

    def get_meaturement(self):
        return self.x + np.sqrt(self.mes_noise_var)*np.random.randn()

    def true_state(self):
        return np.array([[self.v],[self.x]])
