
from rover import Rover
from fastslam import FASTSLAM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ts = 0.1
num_st = 3
plot_movement = True#False#
x0 = np.array([[-5], [-3], [np.pi/2]])

num_lm = 50
lmloc = 20*np.random.rand(2*num_lm) - 10
# lmloc = np.random.randint(-10,10, size=2*num_lm)
landmarks = zip(lmloc[0::2],lmloc[1::2])

if __name__ == '__main__':
    rov = Rover(x0=x0, lm=landmarks)
    fastslam = FASTSLAM(num_lm=num_lm)
    truth = np.array([]).reshape(num_st,0)
    estimate = np.array([]).reshape(num_st,0)
    # variance = np.array([]).reshape(num_st,0)
    gain = np.array([]).reshape(num_st,0)
    time = np.arange(0, 30, ts)

    plt.figure(0)
    plt.ion()

    i = 0
    for t in time:
        u = np.array([[1 + 0.5*np.cos(2*np.pi*0.2*t)], [-0.2 + 2*np.cos(2*np.pi*0.6*t)]])

        if plot_movement and time.tolist().index(t)%10 == 0:
            plt.figure(0)
            plt.clf()
            plt.axis([-10, 10, -10, 10])
            [[x],[y],[th]] = rov.true_state()
            plt.scatter(x, y) # true rover possition
            plt.scatter(x + 0.5*np.cos(th), y + 0.5*np.sin(th), color='r') # true rover heading
            plt.scatter(lmloc[0::2],lmloc[1::2]) # true landmark locations
            mes,ids = rov.get_mesurement()
            r = mes[0::2,0]
            b = mes[1::2,0]
            plt.scatter(x + r*np.cos(th+b), y + r*np.sin(th+b), color='g') # rover measurements in true robot frame
            st,na = fastslam.est_state()
            st = x0 + np.array([[np.cos(x0[2,0]),-np.sin(x0[2,0]),0],[np.sin(x0[2,0]),np.cos(x0[2,0]),0], [0,0,1]]).dot(st)
            plt.scatter(st[0,0],st[1,0], color='m') # estimated robot state rotated into world frame
            plt.scatter(st[0,0] + 0.5*np.cos(st[2,0]), st[1,0] + 0.5*np.sin(st[2,0]), color='m') # estimtated rover heading
            plt.scatter(st[0,0] + r*np.cos(st[2,0]+b), st[1,0] + r*np.sin(st[2,0]+b), color='y') # rover measurements in estimated robot frame

            plt.figure(1)
            plt.clf()
            a = plt.subplot(111, aspect='equal')
            states = fastslam.all_states()
            for i in range(len(states)):
                p = x0[0:2,:] + np.array([[np.cos(x0[2,0]),-np.sin(x0[2,0])],[np.sin(x0[2,0]),np.cos(x0[2,0])]]).dot(states[i])
                e = Ellipse((p[0,0], p[1,0]), 0.2, 0.2, 0)
                a.add_artist(e)
                e.set_alpha(0.5)
                e.set_facecolor([1.,0.5,0.5])

            e = Ellipse((x,y),0.15,0.15,0)
            a.add_artist(e)
            e.set_facecolor([1.,0.,0.])

            loc, other = fastslam.lm_elps()
            for i in range(loc.shape[0]):
                p = loc[i:i+1,:].T
                p = x0[0:2,:] + np.array([[np.cos(x0[2,0]),-np.sin(x0[2,0])],[np.sin(x0[2,0]),np.cos(x0[2,0])]]).dot(p)
                e = Ellipse((p[0,0] , p[1,0]), other[i,0], other[i,1], (other[i,2] + x0[2,0])*180/np.pi)
                a.add_artist(e)
                e.set_alpha(0.3)

            for lm in landmarks:
                e = Ellipse(lm, 0.15, 0.15, 0)
                a.add_artist(e)
                e.set_facecolor([0.,1.0,0.])

            plt.xlim(-10, 10)
            plt.ylim(-10, 10)

            plt.pause(0.00001)
            # plt.pause(5)

        i += 1

        rov.propogate_dynamics(u, ts)
        truth = np.concatenate((truth, rov.true_state()),axis=1)

        fastslam.propogate(u, ts)
        mes,ids = rov.get_mesurement()
        fastslam.update(mes,ids)

        st,na = fastslam.est_state()
        st = x0 + np.array([[np.cos(x0[2,0]),-np.sin(x0[2,0]),0],[np.sin(x0[2,0]),np.cos(x0[2,0]),0], [0,0,1]]).dot(st)
        estimate = np.concatenate((estimate, st), axis=1)

        fastslam.resample()
        # variance = np.concatenate((variance, ekfslam.vars()), axis=1)

    plt.ioff()

    plt.figure(2)
    plt.subplot(311)
    plt.title("Truth vs Estimate")
    plt.plot(time, truth[0,:], label='truth')
    plt.plot(time, estimate[0,:], label='estimate')
    plt.legend()
    # plt.xlabel("time")
    plt.ylabel("x")
    plt.subplot(312)
    plt.plot(time, truth[1,:], label='truth')
    plt.plot(time, estimate[1,:], label='estimate')
    # plt.legend()
    # plt.xlabel("time")
    plt.ylabel("y")
    plt.subplot(313)
    plt.plot(time, truth[2,:], label='truth')
    plt.plot(time, estimate[2,:], label='estimate')
    # plt.legend()
    plt.xlabel("Time")
    plt.ylabel("theta (rad)")

    # plt.figure(3)
    # plt.subplot(311)
    # plt.title("Truth vs Estimate")
    # plt.plot(time, truth[0,:] - estimate[0,:], label='error')
    # plt.plot(time, 2*variance[0,:], label='std div (2sig)')
    # plt.plot(time, -2*variance[0,:], label='std div (2sig)')
    # plt.legend()
    # # plt.xlabel("time")
    # plt.ylabel("x")
    # plt.subplot(312)
    # plt.plot(time, truth[1,:] - estimate[1,:], label='error')
    # plt.plot(time, 2*variance[1,:], label='std div (2sig)')
    # plt.plot(time, -2*variance[1,:], label='std div (2sig)')
    # # plt.legend()
    # # plt.xlabel("time")
    # plt.ylabel("y")
    # plt.subplot(313)
    # plt.plot(time, truth[2,:] - estimate[2,:], label='error')
    # plt.plot(time, 2*variance[2,:], label='std div (2sig)')
    # plt.plot(time, -2*variance[2,:], label='std div (2sig)')
    # # plt.legend()
    # plt.xlabel("Time")
    # plt.ylabel("theta (rad)")

    # plt.figure(3)
    # plt.subplot(311)
    # plt.title("Kalman Gain")
    # plt.plot(time, gain[0,:])
    # plt.ylabel("x gain")
    # plt.subplot(312)
    # plt.plot(time, gain[1,:])
    # plt.ylabel("y gain")
    # plt.subplot(313)
    # plt.plot(time, gain[2,:])
    # plt.xlabel("Time")
    # plt.ylabel("theta gain")

    plt.show()

    # import scipy.io as spio
    #
    # kf = KF(ts)
    # mat = spio.loadmat('hw1_soln_data.mat')
    # vtr = mat['vtr']
    # xtr = mat['xtr']
    # t = mat['t']
    # kf.x = mat['mu0']
    # kf.P = mat['Sig0']
    # u = mat['u']
    # z = mat['z']
    #
    # estimate = np.array([]).reshape(2,0)
    # variance = np.array([]).reshape(2,0)
    # gain = np.array([]).reshape(2,0)
    #
    # for i in range(t.shape[1]):
    #     kf.propodate(u[0,i])
    #     gain =  np.concatenate((gain, kf.update(z[0,i])), axis=1)
    #     estimate = np.concatenate((estimate, kf.est_state()), axis=1)
    #     variance = np.concatenate((variance, kf.vars()), axis=1)
    #
    # plt.figure(1)
    # plt.subplot(211)
    # plt.title("States")
    # plt.plot(t[0,:], xtr[0,:], label='truth')
    # plt.plot(t[0,:], estimate[1,:], label='estimate')
    # plt.legend()
    # # plt.xlabel("Time")
    # plt.ylabel("Possition (m)")
    # plt.subplot(212)
    # plt.plot(t[0,:], vtr[0,:], label='truth')
    # plt.plot(t[0,:], estimate[0,:], label='estimate')
    # # plt.legend()
    # plt.xlabel("Time")
    # plt.ylabel("Velocity (m/s)")
    #
    # plt.figure(2)
    # plt.subplot(211)
    # plt.title("Error")
    # plt.plot(t[0,:], np.abs(xtr[0,:] - estimate[1,:]), label='error')
    # plt.plot(t[0,:], 3*variance[1,:], label='std div (3sig)')
    # plt.legend()
    # # plt.xlabel("Time")
    # plt.ylabel("Possition (m)")
    # plt.subplot(212)
    # plt.plot(t[0,:], np.abs(vtr[0,:] - estimate[0,:]), label='error')
    # plt.plot(t[0,:], 3*variance[0,:], label='vel cov (3sig)')
    # # plt.legend()
    # plt.xlabel("Time")
    # plt.ylabel("Velocity (m/s)")
    #
    # plt.figure(3)
    # plt.subplot(211)
    # plt.title("Kalman Gain")
    # plt.plot(t[0,:], gain[1,:])
    # plt.ylabel("Possition gain")
    # plt.subplot(212)
    # plt.plot(t[0,:], gain[0,:])
    # plt.xlabel("Time")
    # plt.ylabel("Velocity gain")
    #
    # plt.show()
