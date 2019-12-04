
from rover import Rover
from ekf import EKF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ts = 0.1
num_st = 3
x0 = np.array([[-5], [-3], [np.pi/2]])

if __name__ == '__main__':
    rov = Rover()
    ekf = EKF()
    truth = np.array([]).reshape(num_st,0)
    estimate = np.array([]).reshape(num_st,0)
    variance = np.array([]).reshape(num_st,0)
    gain = np.array([]).reshape(num_st,0)
    time = np.arange(0, 20, ts)

    # plt.figure(0)
    # plt.axis([-10, 10, -10, 10])
    # plt.scatter([6,-7,6],[4,8,-4])
    plt.ion()

    for t in time:
        #u = np.array([[1 + 0.5*np.cos(2*np.pi*0.2*t)], [-0.2 + 2*np.cos(2*np.pi*0.6*t)]])
        u = np.array([[1.5], [-0.3]])

        rov.propogate_dynamics(u, ts)
        truth = np.concatenate((truth, rov.true_state()),axis=1)

        # plt.figure(0)
        # plt.clf()
        # plt.axis([-10, 10, -10, 10])
        [[x],[y],[t]] = rov.true_state()
        # plt.scatter(x, y)
        # plt.scatter(x + 0.5*np.cos(t), y + 0.5*np.sin(t), color='r')
        # mes = rov.get_mesurement()
        # for i in range(3):
        #     r = mes[2*i,0]
        #     b = mes[2*i + 1,0]
        #     plt.scatter(x + r*np.cos(t+b), y + r*np.sin(t+b), color='g')
        # plt.scatter([6,-7,6],[4,8,-4])


        plt.figure(1)
        plt.clf()
        a = plt.subplot(111, aspect='equal')
        st = ekf.est_state()
        e1, e2, ang = ekf.state_elps()
        e = Ellipse((st[0,0], st[1,0]), e1, e2, (ang + x0[2,0])*180/np.pi)
        a.add_artist(e)
        e.set_alpha(0.5)
        e.set_facecolor([1.,0.,0.])

        e = Ellipse((x,y),0.15,0.15,0)
        a.add_artist(e)
        e.set_facecolor([1.,0.,0.])

        plt.plot(truth[0,:], truth[1,:])
        plt.plot(estimate[0,:], estimate[1,:])

        plt.axis([-10, 10, -10, 10])

        plt.pause(0.00001)

        ekf.propogate(u, ts)
        # gain =  np.concatenate((gain, ekf.update(rov.get_mesurement())), axis=1)
        estimate = np.concatenate((estimate, ekf.est_state()), axis=1)
        variance = np.concatenate((variance, ekf.vars()), axis=1)

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
    plt.plot(time, 2*variance[0,:], 'r', label='2 sig')
    plt.plot(time, -2*variance[0,:], 'r', label='2 sig')
    plt.legend()
    # plt.xlabel("time")
    plt.ylabel("x")
    plt.subplot(312)
    plt.plot(time, np.abs(truth[1,:] - estimate[1,:]), label='error')
    plt.plot(time, 2*variance[1,:], 'r', label='2 sig')
    plt.plot(time, -2*variance[1,:], 'r', label='2 sig')
    # plt.legend()
    # plt.xlabel("time")
    plt.ylabel("y")
    plt.subplot(313)
    plt.plot(time, np.abs(truth[2,:] - estimate[2,:]), label='error')
    plt.plot(time, 2*variance[2,:], 'r', label='2 sig')
    plt.plot(time, -2*variance[2,:], 'r', label='2 sig')
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
