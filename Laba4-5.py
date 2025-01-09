import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


# Генерация данных для выборки и данных для восстановления плотности распределения вероятности 
np.random.seed(42)
data = np.concatenate([
    np.random.normal(loc=-2, scale=1.5, size=250),
    np.random.normal(loc=2, scale=0.5, size=250)
])

x_distribution = np.linspace(*(data.min() - 1, data.max() + 1), 1000)


# Реализация метода ядерного сглаживания 
def kernel_density_estimation(data, bandwidth):
    kde = gaussian_kde(data, bw_method=bandwidth)
    return kde

kde = kernel_density_estimation(data, bandwidth=0.5)
y_kde = kde(x_distribution)


# Реализация ЕМ алгоритма 
x_distribution = np.linspace(*(data.min() - 1, data.max() + 1), 1000)[:, None] # преобразуем в двумерный массив

def em_density_estimation(data, n_components=2):
    data = data[:, None] # преобразуем в двумерный массив
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data)
    return gmm

gmm = em_density_estimation(data, n_components=2)
log_density = gmm.score_samples(x_distribution)
y_gmm = np.exp(log_density)


plt.figure(figsize=(12, 8))
plt.hist(data, bins=30, density=True, alpha=0.5, label='Гистограмма', color='grey')
plt.plot(x_distribution, y_kde, label='KDE (Kernel Density Estimation)', color='red')
plt.plot(x_distribution, y_gmm, label='EM (Expectation-Maximization) Algorithm', color='green')
plt.grid(True)
plt.legend()
plt.title('Восстановление плотности вероятности методом ядерного сглаживания и ЕМ алгоритмом')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.show()


n_samples = 1000


# Реализация метода Метрополиса-Гастингса 
def metropolis_hastings(target_density, init_x, n_samples, proposal_std):
    samples = [init_x]
    current = init_x
    for _ in range(n_samples - 1):
        proposal = np.random.normal(current, proposal_std)
        acceptance_ratio = target_density(proposal) / target_density(current)
        if np.random.rand() < acceptance_ratio:
            current = proposal
        samples.append(current)
    return np.array(samples)


# Реализация метода Гиббса 
def gibbs_sampling(init_x_y, conditional_distributions, n_samples):
    samples = [init_x_y]
    current = np.array(init_x_y)
    for _ in range(n_samples - 1):
        for i, conditional in enumerate(conditional_distributions):
            current[i] = conditional(*current[:i], *current[i + 1:])
        samples.append(current.copy())
    return np.array(samples)


# Применение методов МГ и Гиббса к плотности, восстановленной KDE 
target_density = kde.evaluate

mh_samples = metropolis_hastings(target_density, data.mean(), n_samples, proposal_std=1.0)
gibbs_samples = gibbs_sampling(
    [data.mean()],
    [lambda: np.random.choice(data)],
    n_samples
)

plt.figure(figsize=(12, 8))
plt.plot(x_distribution, y_kde, label='KDE (Kernel Density Estimation)', color='red')
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, label='Данные Метрополиса-Гастинга', color='cyan')
plt.hist(gibbs_samples, bins=30, density=True, alpha=0.5, label='Данные Гиббса', color='orange')
plt.title('Методы МГ и Гиббса на плотности, восстановленной методом ядерного сглаживания')
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.show()


# Применение методов МГ и Гиббса к плотности, восстановленной EM алгоритмом 
def target_density(x):
    return np.exp(gmm.score_samples(np.array([[x]])))

mh_samples = metropolis_hastings(target_density, data.mean(), n_samples, proposal_std=1.0)
gibbs_samples = gibbs_sampling(
    [data.mean()],
    [lambda: np.random.choice(data)],
    n_samples
)

plt.figure(figsize=(12, 8))
plt.plot(x_distribution, y_gmm, label='EM (Expectation-Maximization) Algorithm', color='green')
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, label='Данные Метрополиса-Гастинга', color='cyan')
plt.hist(gibbs_samples, bins=30, density=True, alpha=0.5, label='Данные Гиббса', color='orange')
plt.title('Методы МГ и Гиббса на плотности, восстановленной ЕМ алгоритмом')
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.show()


# Реализация блуждания в случае трехмерной функции плотности в методе МГ 
def target_density_3d(x, y, z):
    return np.exp(-0.5 * (x ** 2 + y ** 2 + z ** 2)) # mu = 0, sigma = 1


# Метод МГ для трехмерного набора точек 
def metropolis_hastings_3d(target_density, init_point, n_samples, proposal_std):
    samples = [init_point]
    current = np.array(init_point)
    for _ in range(n_samples - 1):
        proposal = current + np.random.normal(0, proposal_std, size=3)
        acceptance_ratio = target_density(*proposal) / target_density(*current)
        if np.random.rand() < acceptance_ratio:
            current = proposal
        samples.append(current.copy())
    return np.array(samples)

n_samples_3d = 5000
init_point_3d = [0, 0, 0]
proposal_std_3d = 1.0

samples_3d = metropolis_hastings_3d(target_density_3d, init_point_3d, n_samples_3d, proposal_std_3d)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples_3d[:, 0], samples_3d[:, 1], samples_3d[:, 2], c='blue', alpha=0.5, s=1)
ax.set_title('Метод Метрополиса-Гастингса для трехмерной плотности распределения вероятности')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# Сравнение красного и синего набора точек (исходного и сгенерированного) 
red_dots = np.random.normal(loc=0, scale=1.0, size=500)

def target_density_1d(x):
    return np.exp(-0.5 * x ** 2)

blue_dots = metropolis_hastings(target_density_1d, init_x=0, n_samples=1000, proposal_std=1.0)

red_kde = gaussian_kde(red_dots)
blue_kde = gaussian_kde(blue_dots)


def kl_divergence(p, q, grid):
    p_vals = p(grid)
    q_vals = q(grid)
    return np.sum(p_vals * np.log(p_vals / q_vals)) * (grid[1] - grid[0])


grid = np.linspace(min(red_dots.min(), blue_dots.min()), max(red_dots.max() + 1, blue_dots.max()) + 1, 1000)


kl_red_to_blue = kl_divergence(red_kde, blue_kde, grid)
kl_blue_to_red = kl_divergence(blue_kde, red_kde, grid)


print(f"KL-дивергенция (исходный -> сгенерированный): {kl_red_to_blue:.5f}")
print(f"KL-дивергенция (сгенерированный -> исходный): {kl_blue_to_red:.5f}")


plt.figure(figsize=(12, 8))
plt.plot(grid, red_kde(grid), label='Исходный набор (красный)', color='red')
plt.plot(grid, blue_kde(grid), label='Сгенерированный набор (синий)', color='blue')
plt.fill_between(grid, 0, red_kde(grid), color='red', alpha=0.2)
plt.fill_between(grid, 0, blue_kde(grid), color='blue', alpha=0.2)
plt.title('Сравнение исходного и сгенерированного набора')
plt.grid(True)
plt.legend()
plt.show()