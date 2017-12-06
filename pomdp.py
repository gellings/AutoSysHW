import numpy as np
import matplotlib.pyplot as plt
# from collections import set

p_z = np.array([[0.7, 0.3], # p(z1) = 0.7*p1 + 0.3(1 - p1) same for z2
                 [0.3, 0.7]])
p_t = np.zeros((2,2,3))
p_t[:,:,2] = np.array([[0.8, 0.2], # p(x1|u3,x2) = 0.8*p1 + 0.2(1 - p1)
                [0.2, 0.8]])
r = np.array([[-100, 100, -1],
              [100, -50, -1]]) # r(b,u1) = -100*p1 + 100(1 - p1)

if __name__ == '__main__':

    t_horz = 4
    N = 2 # number of states
    Nu = 3 # number of control actions
    Nm = 2 # number of measurements
    gamma = 1

    Yuv = np.array([[0, 0, 0]])

    for t in range(t_horz):
        Yuv_p = set() #using a set to remove of duplicates
        vv = np.empty((Yuv.shape[0],Nu,Nm,N))
        for k in range(Yuv.shape[0]):
            for u in range(Nu):
                for z in range(Nm):
                    for j in range(N):
                        vk1 = Yuv[k,1]*p_z[0,z]*p_t[0,j,u]
                        vk2 = Yuv[k,2]*p_z[1,z]*p_t[1,j,u]
                        vv[k,u,z,j] = vk1 + vk2
        for u in range(Nu):
            for k1 in range(Yuv.shape[0]):
                for k2 in range(Yuv.shape[0]):
                    vpr = [0,0]
                    for i in range(N):
                        vpr[i] = gamma*(r[i,u] + vv[k1,u,0,i] + vv[k2,u,1,i])
                    if vpr[0] > 0 or vpr[1] > 0: # this is my pruning
                        vpr = np.concatenate(([u],vpr))
                        Yuv_p.add(tuple(vpr.tolist()))

        # prune?
        Yuv_p = [list(elem) for elem in Yuv_p]
        Yuv = np.array(Yuv_p)

    print Yuv
    # plot
    plt.figure(0)
    # for i in range(Yuv.shape[0]):
        # plt.plot([1,0], Yuv[i,1:], 'blue')
    plt.plot(np.array([[1,0]]*Yuv.shape[0]).T, Yuv[:,1:].T, 'blue')
    plt.show()

    polocy = [0.205, 0.701]
    p1_0 = 0.6
    x_0 = 0
    r_0 = 0

    rs = np.array([])
    for i in range(100):
        p1 = p1_0
        x = x_0
        r = r_0
        while p1 > polocy[0] and p1 < polocy[1]:

            if np.random.rand() < 0.8: # correct transition
                x = int(not x)
            # acount for u3 transition
            # p(x1|u3,x2) = 0.8*p1 + 0.2(1 - p1)
            p1 = 0.2*p1 + 0.8*(1 - p1)

            # use measurement
            if np.random.rand() < 0.7: # correct measurement
                p1 = 0.7*p1/(0.4*p1 + 0.3)
            else:
                p1 = 0.3*p1/(-0.4*p1 + 0.7)

            # get the reward for u3
            r -= 1

        # now for the end game
        if p1 < polocy[0]:
            if x == 2:
                r += 100
            else:
                r -= 100
        if p1 > polocy[1]:
            if x == 1:
                r += 100
            else:
                r -= 50
        print r
        rs = np.concatenate((rs,[r]))
    print 'average reward', np.mean(rs)
