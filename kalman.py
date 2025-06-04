import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

def kalman_smooth(data):
    N = len(data)
    dt = 1  # 每帧时间间隔

    # 初始化 Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.R *= 1.0     # 观测噪声协方差
    kf.Q *= 0.01    # 过程噪声协方差
    kf.P *= 10.0    # 初始状态协方差
    kf.x = np.array([data[0,1], data[0,2], 0, 0])  # 初始状态

    smoothed = []

    for i in range(N):
        confidence, x_meas, y_meas = data[i]

        # 预测
        kf.predict()

        if confidence:
            z = np.array([x_meas, y_meas])
            kf.update(z)

        # 记录当前估计的 x, y
        smoothed.append([1, kf.x[0], kf.x[1]])
        
    ret = np.array(smoothed)
    # plt.figure()
    # plt.plot(data[:, 1], data[:, 2], marker='*')
    # plt.plot(ret[:, 1], ret[:, 2], marker='o')
    return ret
