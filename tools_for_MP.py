import numpy as np
import matplotlib.pyplot as plt
import math
import random

def plot_particles_history(particles_history, times):
    states = [f'{i}' for i in range(particles_history.shape[0])]
    n = particles_history.shape[1]
    for i in range(n):
        plt.plot(times, particles_history[:, i], label=f"{states[i]}")
    plt.xlabel("time (s)")
    plt.ylabel("particles")
    plt.legend()
    plt.show()

def inverse_exp(random_, coef):
    '''возвращает значение случайной величины,
    распределённой по экспоненциальному закону,
    соответствующему простейшему Пуассоновскому потоку'''
    if coef == 0:
        return np.finfo(np.float32).max
    return -1/coef * math.log(1 - random_)

def update_tau(tau, phi):
    ''' Возвращает матрицу с временами событий на следующем шаге и ближайшее событие
    :param tau: array of trigger times, generated on the previous step
    :return: times array on the next step, next event, time of next event'''
    min_ = np.finfo(np.float32).max
    idx_min = 0

    for k in range(len(tau)):
        if tau[k] == np.finfo(np.float32).max and phi[k] != 0:
            tau[k] = inverse_exp(random.random(), phi[k])
        if tau[k] < min_:
            min_ = tau[k]
            idx_min = k
    tau -= min_
    return tau, idx_min, min_

def update_phi(alpha, eps, phi):
    phi = phi
    phi[~np.all(eps <= alpha, axis=1)] = 0
    return phi

def taking_action(alpha, next_event, eps, p_gamma):
    alpha += -eps[next_event] + p_gamma[next_event]
    phi = update_phi(alpha)
    return alpha, phi

