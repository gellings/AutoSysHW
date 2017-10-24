import numpy as np

class OGM:
    def __init__(self, thk, alpha=1.2, beta=1.0, z_max=150.0, l_0=0.5, l_occ=0.65, l_free=0.35):
        self.map = l_0 + np.zeros((100,100))
        self.thk = thk
        self.alpha = alpha
        self.beta = beta*np.pi/180
        self.z_max = z_max
        self.l_0 = l_0
        self.l_occ = l_occ
        self.l_free = l_free

    def update_map(self, x, z):
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                m = np.array([[i],[j]])
                zm = z.copy()
                zm[np.isnan(z)] = self.z_max
                zm = np.concatenate((zm, self.thk), axis=0)
                self.map[i,j] += self.inv_range_model(m,x,zm) - self.l_0

        self.map = np.clip(self.map,0,1)

    def get_map(self):
        return self.map

    def inv_range_model(self, m, x, z):
        r = np.linalg.norm(m - x[0:2,:])
        phi = np.arctan2(m[1,0]-x[1,0],m[0,0]-x[0,0]) - x[2,0]
        if phi > np.pi:
            phi -= 2*np.pi
        if phi < -np.pi:
            phi += 2*np.pi
        k = np.argmin(np.abs(phi - z[1,:]))
        if r > min(self.z_max, z[0,k] + self.alpha/2) or np.abs(phi - z[1,k]) > self.beta/2:
            return self.l_0
        if z[0,k] < self.z_max and np.abs(r - z[0,k]) < self.alpha/2:
            return self.l_occ
        if r <= z[0,k]:
            return self.l_free
        print "warn"
        return self.l_0
