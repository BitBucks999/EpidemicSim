import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import streamlit as st
import matplotlib.animation as animation
import io
import imageio.v2 as imageio
from matplotlib.animation import PillowWriter
import matplotlib.colors as mcolors



# Функция для модели SIR
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Функция для модели SEIR
def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Основная функция приложения
def main():
    st.title('Моделирование эпидемии')

    # Выбор модели
    model_type = st.selectbox("Выберите модель", ["SIR", "SEIR"])

    # Ввод параметров
    N = st.number_input("Численность популяции", value=1000)
    I0 = st.number_input("Начальное количество зараженных", value=10)
    R0 = st.number_input("Начальное количество выздоровевших", value=0)
    E0 = 0
    S0 = N - I0 - R0
    beta = st.slider("Коэффициент заражения (β)", min_value=0.0, max_value=1.0, value=0.3)
    gamma = st.slider("Скорость выздоровления (γ)", min_value=0.0, max_value=1.0, value=0.1)
    sigma = 1/5.2 if model_type == "SEIR" else None
    if model_type == "SEIR":
        sigma = st.slider("Среднее время инкубации (1/σ)", min_value=0.0, max_value=1.0, value=1/5.2)

    # Время симуляции
    t = np.linspace(0, 50, 50)

    # Начальные условия
    if model_type == "SIR":
        initial_conditions = [S0, I0, R0]
        solution = odeint(sir_model, initial_conditions, t, args=(N, beta, gamma))
        S, I, R = solution.T
    else:
        initial_conditions = [S0, E0, I0, R0]
        solution = odeint(seir_model, initial_conditions, t, args=(N, beta, sigma, gamma))
        S, E, I, R = solution.T

    # Построение графиков
    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', label='Восприимчивые')
    if model_type == "SEIR":
        ax.plot(t, E, 'y', label='Подвергшиеся')
    ax.plot(t, I, 'r', label='Зараженные')
    ax.plot(t, R, 'g', label='Выздоровевшие')
    ax.set_xlabel('Время (дни)')
    ax.set_ylabel('Число людей')
    ax.legend()
    ax.grid()

    # Отображение графиков в Streamlit
    st.pyplot(fig)

    # Параметры для симуляции взаимодействий
    st.subheader("Симуляция взаимодействий")
    grid_size = st.slider("Размер сетки", min_value=10, max_value=200, value=50)
    infection_radius = st.slider("Радиус заражения", min_value=1, max_value=5, value=1)
    infection_probability = st.slider("Вероятность заражения при контакте", min_value=0.0, max_value=1.0, value=0.2)
    initial_infected = st.number_input("Начальное количество зараженных агентов", value=10)
    days = st.slider("Длительность симуляции (дни)", min_value=1, max_value=365, value=50)

    if st.button("Запустить симуляцию"):
        run_simulation(grid_size, N, initial_infected, infection_radius, infection_probability, days)

# Функция для запуска симуляции
def run_simulation(grid_size, population_size, initial_infected, infection_radius, infection_probability, days):
    grid = np.zeros((grid_size, grid_size))
    agents = []

    class Agent:
        def __init__(self, state):
            self.state = state
            self.x = np.random.randint(0, grid_size)
            self.y = np.random.randint(0, grid_size)
            self.days_infected = 0
        
        def move(self):
            self.x = (self.x + np.random.choice([-1, 0, 1])) % grid_size
            self.y = (self.y + np.random.choice([-1, 0, 1])) % grid_size

        def infect(self):
            for dx in range(-infection_radius, infection_radius + 1):
                for dy in range(-infection_radius, infection_radius + 1):
                    if (0 <= self.x + dx < grid_size) and (0 <= self.y + dy < grid_size):
                        if grid[self.x + dx, self.y + dy] == 0:
                            if np.random.random() < infection_probability:
                                grid[self.x + dx, self.y + dy] = 1
                                
        def recover(self, recovery_time=10):
            """Процесс выздоровления"""
            if self.state == 1:
               self.days_infected += 1
               if self.days_infected >= recovery_time:  # После определенного времени заражения агент выздоравливает
                  self.state = 2  # Выздоровевший агент

                                

    # Инициализация агентов
    for _ in range(population_size - initial_infected):
        agents.append(Agent(state=0))  # Здоровые агенты
    for _ in range(initial_infected):
        agents.append(Agent(state=1))  # Инфицированные агенты

    # Размещение агентов на сетке
    for agent in agents:
        grid[agent.x, agent.y] = agent.state

    # Подготовка для анимации
    fig, ax = plt.subplots()
    cmap = mcolors.ListedColormap(['white', 'white', 'red'])
    bounds = [-1, 0, 1, 2]  # Пределы значений состояний
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(grid, cmap=cmap, norm=norm)
    plt.axis('off')

    # Список кадров для GIF
    gif_frames = []

    # Запуск симуляции
    for _ in range(days):
        new_grid = np.copy(grid)
        for agent in agents:
            agent.move()
            if agent.state == 1:
                agent.infect()
            new_grid[agent.x, agent.y] = agent.state
        grid = new_grid

        im.set_array(grid)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        gif_frames.append(imageio.imread(buf))
        buf.close()

    # Создание GIF в памяти
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, gif_frames, format='GIF', fps=10)
    gif_buf.seek(0)

    # Отображение GIF в Streamlit
    st.image(gif_buf, caption="Анимация эпидемии")


# Запуск основного приложения
if __name__ == "__main__":
    main()
