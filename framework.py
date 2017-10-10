
from rover import Rover
from pf import PF
import numpy as np
import matplotlib.pyplot as plt

ts = 0.1
num_st = 3
plot_movement = True

if __name__ == '__main__':
    rov = Rover()
    pf = PF()
    truth = np.array([]).reshape(num_st,0)
    estimate = np.array([]).reshape(num_st,0)
    variance = np.array([]).reshape(num_st,0)
    gain = np.array([]).reshape(num_st,0)
    time = np.arange(0, 20, ts)

    plt.figure(0)
    plt.axis([-10, 10, -10, 10])
    plt.scatter([6,-7,6],[4,8,-4])
    plt.ion()

    for t in time:
        u = np.array([[1 + 0.5*np.cos(2*np.pi*0.2*t)], [-0.2 + 2*np.cos(2*np.pi*0.6*t)]])

        rov.propogate_dynamics(u, ts)
        truth = np.concatenate((truth, rov.true_state()),axis=1)

        if plot_movement:
            plt.clf()
            plt.axis([-10, 10, -10, 10])
            [[x],[y],[t]] = rov.true_state()
            plt.scatter(x, y)
            plt.scatter(x + 0.5*np.cos(t), y + 0.5*np.sin(t), color='r')
            mes = rov.get_mesurement()
            for i in range(3):
                r = mes[2*i,0]
                b = mes[2*i + 1,0]
                plt.scatter(x + r*np.cos(t+b), y + r*np.sin(t+b), color='g')
            chi = pf.get_chi()
            plt.scatter(chi[:,0],chi[:,1], color='m')
            plt.scatter([6,-7,6],[4,8,-4])
            plt.pause(0.00001)

        pf.propodate(u, ts)
        pf.update(rov.get_mesurement())
        estimate = np.concatenate((estimate, pf.est_state()), axis=1)
        variance = np.concatenate((variance, pf.vars()), axis=1)

    plt.ioff()

    plt.figure(1)
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

    plt.figure(2)
    plt.subplot(311)
    plt.title("Truth vs Estimate")
    plt.plot(time, truth[0,:] - estimate[0,:], label='error')
    plt.plot(time, 2*variance[0,:], label='std div (2sig)')
    plt.plot(time, -2*variance[0,:], label='std div (2sig)')
    plt.legend()
    # plt.xlabel("time")
    plt.ylabel("x")
    plt.subplot(312)
    plt.plot(time, truth[1,:] - estimate[1,:], label='error')
    plt.plot(time, 2*variance[1,:], label='std div (2sig)')
    plt.plot(time, -2*variance[1,:], label='std div (2sig)')
    # plt.legend()
    # plt.xlabel("time")
    plt.ylabel("y")
    plt.subplot(313)
    plt.plot(time, truth[2,:] - estimate[2,:], label='error')
    plt.plot(time, 2*variance[2,:], label='std div (2sig)')
    plt.plot(time, -2*variance[2,:], label='std div (2sig)')
    # plt.legend()
    plt.xlabel("Time")
    plt.ylabel("theta (rad)")

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
