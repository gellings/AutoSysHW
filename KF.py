import numpy as np

class KF:
    def __init__(self, ts):
        m = 100 #kg
        b = 20 #N-s/m
        self.x = np.zeros((2,1))
        self.P = np.zeros((2,2))
        self.A = np.eye(2) + ts*np.array([[-b/m, 0], [1, 0]])
        self.Q = np.array([[0.01, 0],
                           [0, 0.0001]])

        self.B = np.array([[1/m], [0]])

        self.C = np.array([[0, 1]])
        self.R = np.array([[0.001]])

    def propodate(self, u):
        self.x = self.A.dot(self.x) + self.B.dot(u)
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q

    def update(self, z):
        inv = np.linalg.inv(self.C.dot(self.P).dot(self.C.T) + self.R)
        K = self.P.dot(self.C.T).dot(inv)
        self.x = self.x + K.dot(z - self.C.dot(self.x))
        self.P = (np.eye(2) - K.dot(self.C)).dot(self.P)
        return K

    def est_state(self):
        return self.x

    def vars(self):
        return np.array([[np.sqrt(self.P[0,0])], [np.sqrt(self.P[1,1])]])
