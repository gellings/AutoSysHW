import numpy as np
import matplotlib.pyplot as plt

def make_map():
    N = 100
    Np = N + 2
    new_map = np.zeros((Np,Np)) # map dimension

    # Initialize walls and obstacle maps as empty
    walls = np.zeros((Np,Np))
    obs1 = np.zeros((Np,Np))
    obs2 = np.zeros((Np,Np))
    obs3 = np.zeros((Np,Np))
    goal = np.zeros((Np,Np))

    # Create exterior walls
    walls[1,1:N+1] = 1
    walls[1:N+1,1] = 1
    walls[N,1:N+1] = 1
    walls[1:N+1,N] = 1

    # Create single obstacle
    obs1[19:40,29:80] = 1
    obs1[9:20,59:65] = 1

    # Another obstacle
    obs2[44:65,9:45] = 1

    # Another obstacle
    obs3[42:92,74:85] = 1
    obs3[69:80,49:75] = 1

    # The goal states
    goal[74:80,95:98] = 1

    # Put walls and obstacles into map
    new_map = walls + obs1 + obs2 + obs3 + goal

    # Plot map
    # Sort through the cells to determine the x-y locations of occupied cells
    Mm, Nm = new_map.shape
    xm = np.array([]).reshape(0,1)
    ym = np.array([]).reshape(0,1)
    for i in range(1,Mm):
        for j in range(1,Nm):
            if new_map[i,j]:
                xm = np.concatenate((xm, [[i]]))
                ym = np.concatenate((ym, [[j]]))

    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.plot(xm,ym,'.')
    plt.axis('square')
    plt.axis([0, N+1, 0, N+1])

    plt.pause(0.1)
    plt.ioff()

    return new_map

def make_reward_map():
    N = 100
    Np = N + 2
    new_map = -2.0*np.ones((Np,Np)) # map dimension

    # Create exterior walls
    new_map[1,1:N+1] = -100
    new_map[1:N+1,1] = -100
    new_map[N,1:N+1] = -100
    new_map[1:N+1,N] = -100

    # Create single obstacle
    new_map[19:40,29:80] = -5000
    new_map[9:20,59:65] = -5000

    # Another obstacle
    new_map[44:65,9:45] = -5000

    # Another obstacle
    new_map[42:92,74:85] = -5000
    new_map[69:80,49:75] = -5000

    # The goal states
    new_map[74:80,95:98] = 10000

    # Plot map

    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.imshow(new_map.T, cmap='gray')
    plt.axis('square')
    plt.axis([0, N+1, 0, N+1])

    plt.pause(0.1)
    plt.ioff()

    return new_map
