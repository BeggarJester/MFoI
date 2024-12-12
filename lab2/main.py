import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

r_values = [0.1, 0.5, 0.9]


def renyi_entropy(data, epsilon_range, d=2):
    C_d = [c_d(data, epsilon, d) for epsilon in epsilon_range]
    K2_values = [- np.log(C_d[i] / C_d[i + 1]) if C_d[i + 1] > 0 else 0 for i in range(len(C_d) - 1)]

    return K2_values


def c_d(data, epsilon, d):
    # N = len(data)
    # count = 0
    # for i in range(N - d):
    #     for j in range(i + 1, N - d):
    #         if np.linalg.norm(data[i:i + d] - data[j:j + d]) < epsilon:
    #             count += 1
    # return count / (N * (N - d))
    N = len(data)
    count = 0
    for i in range(N - d):
        distances = np.linalg.norm(data[i:i + d] - data[i + 1:N - d + 1][:, np.newaxis], axis=1)
        count += np.sum(distances < epsilon)
    return count / (N * (N - d))


def plot_results(ax, x, y, title):
    ax.plot(x, y, color='black')
    ax.set_title(f'{title}\nK2 = {y[-1]:.2f}')
    ax.set_xlabel('ε')
    ax.set_ylabel('K2')
    ax.grid()


def main(signals_data):
    for idx, signal in enumerate(signals_data):

        data = pd.read_excel(signal)
        time_series, filtered_data = data.iloc[::10, 0].values, data.iloc[::10, 2].values

        fig, axs = plt.subplots(len(r_values), 1, figsize=(10, 15))

        for i, r in enumerate(r_values):
            epsilon_range = np.linspace(0.01, r, 100)
            K2_values = renyi_entropy(data=filtered_data, epsilon_range=epsilon_range)

            plot_results(axs[i], np.linspace(0.01, r, len(K2_values)), K2_values,
                         f'Rényi entropy K2 for r={r}, ')

        plt.suptitle(f'Rényi entropy K2 for different r, signal {idx + 1}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f'F[{idx + 1}]_filter.png')


if __name__ == "__main__":
    signals = ['1.xlsx', '2.xlsx', '3.xlsx']
    main(signals)
