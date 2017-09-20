
from AUV import UUV
from KF import KF
import numpy as np
import matplotlib.pyplot as plt

ts = 0.05

if __name__ == '__main__':
    uuv = UUV()
    kf = KF(ts)
    truth = np.array([]).reshape(2,0)
    estimate = np.array([]).reshape(2,0)
    variance = np.array([]).reshape(2,0)
    gain = np.array([]).reshape(2,0)

    for t in np.arange(0, 50, ts):
        if t < 5:
            F = 50 #N
        elif t >= 25 and t < 30:
            F = -50 #N
        else:
            F = 0 #N

        uuv.propogate_dynamics(F, ts)
        truth = np.concatenate((truth, uuv.true_state()),axis=1)
        kf.propodate(F)
        gain =  np.concatenate((gain, kf.update(uuv.get_meaturement())), axis=1)
        estimate = np.concatenate((estimate, kf.est_state()), axis=1)
        variance = np.concatenate((variance, kf.vars()), axis=1)

    plt.figure(1)
    plt.subplot(211)
    plt.title("States")
    plt.plot(np.arange(0,50,ts), truth[1,:], label='truth')
    plt.plot(np.arange(0,50,ts), estimate[1,:], label='estimate')
    plt.legend()
    # plt.xlabel("Time")
    plt.ylabel("Possition (m)")
    plt.subplot(212)
    plt.plot(np.arange(0,50,ts), truth[0,:], label='truth')
    plt.plot(np.arange(0,50,ts), estimate[0,:], label='estimate')
    # plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Velocity (m/s)")

    plt.figure(2)
    plt.subplot(211)
    plt.title("Error")
    plt.plot(np.arange(0,50,ts), np.abs(truth[1,:] - estimate[1,:]), label='error')
    plt.plot(np.arange(0,50,ts), 3*variance[1,:], label='std div (3sig)')
    plt.legend()
    # plt.xlabel("Time")
    plt.ylabel("Possition (m)")
    plt.subplot(212)
    plt.plot(np.arange(0,50,ts),np.abs(truth[0,:] - estimate[0,:]), label='error')
    plt.plot(np.arange(0,50,ts), 3*variance[0,:], label='vel cov (3sig)')
    # plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Velocity (m/s)")

    plt.figure(3)
    plt.subplot(211)
    plt.title("Kalman Gain")
    plt.plot(np.arange(0,50,ts), gain[1,:])
    plt.ylabel("Possition gain")
    plt.subplot(212)
    plt.plot(np.arange(0,50,ts), gain[0,:])
    plt.xlabel("Time")
    plt.ylabel("Velocity gain")

    plt.show()
