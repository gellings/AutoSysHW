import numpy as np
import matplotlib.pyplot as plt

from MDP_hw_map import make_reward_map, make_map

if __name__ == '__main__':
    reward_map = make_reward_map()
    obstacle_map = make_map()

    value_map = reward_map.copy()
    old_map = -2.0*np.ones((102,102))

    plt.figure(2)
    plt.clf()
    plt.ion()
    num = 0

    while np.sum(np.abs(value_map - old_map)) > 0.1:
        old_map = value_map.copy()

        for i in range(1,100):
            for j in range(1,100):
                if obstacle_map[i,j] == 0.0:
                    right = value_map[i+1,j]*0.8 + value_map[i,j+1]*0.1 + value_map[i,j-1]*0.1
                    up = value_map[i,j+1]*0.8 + value_map[i+1,j]*0.1 + value_map[i-1,j]*0.1
                    left = value_map[i-1,j]*0.8 + value_map[i,j+1]*0.1 + value_map[i,j-1]*0.1
                    down = value_map[i,j-1]*0.8 + value_map[i+1,j]*0.1 + value_map[i-1,j]*0.1
                    value_map[i,j] = reward_map[i,j] + np.max([right,up,left,down])

        if num%10 == 0:
            plt.clf()
            plt.imshow((np.clip(value_map.T,9500,10000)-9500)/500, cmap='gray')
            plt.axis('square')
            plt.axis([0, 101, 0, 101])

            plt.pause(0.001)
        num += 1

    plt.ioff()

    print "finished", num
    plt.imshow((np.clip(value_map.T,9500,10000)-9500)/500, cmap='gray')

    X = np.array([]).reshape(0,1)
    Y = np.array([]).reshape(0,1)
    U = np.array([]).reshape(0,1)
    V = np.array([]).reshape(0,1)
    for i in range(1,100):
        for j in range(1,100):
            if obstacle_map[i,j] == 0.0:
                right = value_map[i+1,j]*0.8 + value_map[i,j+1]*0.1 + value_map[i,j-1]*0.1
                up = value_map[i,j+1]*0.8 + value_map[i+1,j]*0.1 + value_map[i-1,j]*0.1
                left = value_map[i-1,j]*0.8 + value_map[i,j+1]*0.1 + value_map[i,j-1]*0.1
                down = value_map[i,j-1]*0.8 + value_map[i+1,j]*0.1 + value_map[i-1,j]*0.1
                polocy = np.argmax([right,up,left,down])
                X = np.concatenate((X,[[i]]))
                Y = np.concatenate((Y,[[j]]))
                U = np.concatenate((U,[[30*np.cos(polocy*np.pi/2)]]))
                V = np.concatenate((V,[[30*np.sin(polocy*np.pi/2)]]))

    plt.figure(3)
    plt.quiver(X,Y,U,V)
    plt.show()
