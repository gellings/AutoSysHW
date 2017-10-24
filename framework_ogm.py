import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from ogm import OGM

plot_movement = True

if __name__ == '__main__':
    mat = spio.loadmat('state_meas_data.mat')
    thk = mat['thk']
    X = mat['X']
    z = mat['z']

    ogm = OGM(thk)

    plt.figure(0)
    plt.axis([0, 100, 0, 100])
    plt.ion()

    for i in range(X.shape[1]):
        mes = z[:,:,i]
        sta = X[:,i:i+1] - np.array([[1],[1],[0]])
        ogm.update_map(sta,mes[0:1,:])

        if plot_movement and i%25 == 0:
            plt.figure(0)
            plt.clf()
            plt.axis([0, 100, 0, 100])
            [[x],[y],[t]] = sta
            plt.scatter(x, y)
            plt.scatter(x + 0.5*np.cos(t), y + 0.5*np.sin(t), color='r')
            for j in range(11):
                r = mes[0,j]
                b = mes[1,j]
                # b = thk[0,j]
                if not np.isnan(r):
                    plt.scatter(x + r*np.cos(t+b), y + r*np.sin(t+b), color='g')
            plt.figure(1)
            plt.imshow(1 - np.flip(ogm.get_map().T,0), cmap='gray')
            plt.pause(0.00001)

    plt.ioff()

    plt.figure(1)
    plt.imshow(1 - np.flip(ogm.get_map().T,0), cmap='gray')
    plt.show()
