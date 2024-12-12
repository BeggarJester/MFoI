import numpy as np
import matplotlib.pyplot as plt

h = 0.01
g = 9.8
L = 1
nu = 0.0
sigma_w = 0.05
sigma_v = 0.5
kalman_init_state = np.array([0.1, 0.1])
simulation_init_state = np.array([-2, 0])


def jacobian_matrix(theta):
    return np.array([[1, h], [-h * (g / L) * np.cos(theta)[0], 1 - h * nu]])


def integration_step(state, step_size):
    return np.array([state[0] + state[1] * step_size,
                     - (g * step_size / L) * np.sin(state[0]) + (1 - step_size * nu) * state[1]])


def kalman_filter(data, process_noise, measurement_noise, init_state, step_size):
    filteredData = []
    state = init_state.reshape(2, 1)

    H = np.eye(2)
    P = np.eye(2)

    Q = np.eye(2) * process_noise
    R = np.eye(2) * measurement_noise
    for measurement in data:
        measurement = measurement.reshape(2, 1)
        F = np.eye(2) + step_size * jacobian_matrix(state[0])

        state = integration_step(state, step_size)

        P = np.dot(F, np.dot(P, F.T)) + Q

        S = np.dot(H, np.dot(P, H.T)) + R

        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))

        y = measurement - np.dot(H, state)

        state = state + np.dot(K, y)

        I = np.eye(2)
        P = (I - np.dot(K, H)).dot(P)

        filteredData.append(state)

    filteredData = np.array(filteredData)

    return filteredData


def simulate_pendulum(init_state, t_values):
    state = init_state.reshape(2, 1)
    simulated = []

    simulatedData = np.empty((len(t_values), 2))
    simulatedData[0] = init_state

    for i in range(len(t_values)):
        state = integration_step(state, h)
        simulated.append(state)

    simulated = np.array(simulated)
    return simulated


def main():
    data = np.loadtxt('data1.txt')

    t_values = np.arange(0, 10, h)

    simulated_data = simulate_pendulum(simulation_init_state, t_values)

    filtered_data = kalman_filter(data, sigma_w, sigma_v, kalman_init_state, h)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, data[:, 0], label='data', color='red', linewidth=0.7)
    plt.plot(t_values, simulated_data[:, 0], label='simulation', color='blue')
    plt.plot(t_values, filtered_data[:, 0], label='filter', color='black', linewidth=0.9)

    plt.title('Движение маятника')
    plt.xlabel('t')
    plt.ylabel(r'$\theta$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(data[:, 0], data[:, 1], label='data', color='red', linewidth=0.7)
    plt.plot(simulated_data[:, 0], simulated_data[:, 1], label='simulation', color='blue')
    plt.plot(filtered_data[:, 0], filtered_data[:, 1], label='filter', color='black', linewidth=0.9)

    plt.title('Фазовый портрет')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\omega$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
