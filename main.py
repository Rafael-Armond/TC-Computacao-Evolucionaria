import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Carregar os dados dos clientes
clientes = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

# Parâmetros iniciais
max_num_pas = 30  # Número máximo de pontos de acesso
max_distancia = 85  # Máxima distância entre PA e cliente
max_capacidade_pa = 54  # Capacidade máxima de largura de banda de um PA
grid_resolution = 5  # Resolução da malha
pop_size = 100  # Tamanho da população
num_generations = 500  # Número de gerações
mutation_rate = 0.1  # Taxa de mutação
max_penalty = 1000  # Penalidade máxima para excesso de PAs
unattended_penalty = 10000  # Penalidade para cada cliente não atendido

# Inicialização da população
def initialize_individual(num_pas, grid_resolution):
    return np.round(np.random.rand(num_pas, 2) * 400 / grid_resolution) * grid_resolution

population = [initialize_individual(random.randint(1, max_num_pas), grid_resolution) for _ in range(pop_size)]

best_fitness = -np.inf
best_individual = None
best_fitness_values = []  # Armazenar o melhor fitness de cada geração

# Função de fitness
def evaluate_fitness(individual, clientes, max_distancia, max_capacidade_pa, max_num_pas, max_penalty, unattended_penalty):
    num_pas = individual.shape[0]
    pa_largura_banda_usada = np.zeros(num_pas)
    clientes_atendidos = 0
    clientes_nao_atendidos = 0

    for i in range(clientes.shape[0]):
        distancias = np.sqrt((individual[:, 0] - clientes.iloc[i, 0]) ** 2 + (individual[:, 1] - clientes.iloc[i, 1]) ** 2)
        min_dist = np.min(distancias)
        pa_index = np.argmin(distancias)
        if min_dist <= max_distancia and pa_largura_banda_usada[pa_index] + clientes.iloc[i, 2] <= max_capacidade_pa:
            pa_largura_banda_usada[pa_index] += clientes.iloc[i, 2]
            clientes_atendidos += 1
        else:
            clientes_nao_atendidos += 1

    # Penalizar clientes não atendidos
    fitness = clientes_atendidos - max_penalty * (num_pas - max_num_pas) - unattended_penalty * clientes_nao_atendidos
    return fitness, pa_largura_banda_usada

# Seleção por torneio
def tournament_selection(population, fitness_values, tournament_size=3):
    selected_idx = np.random.randint(0, len(population), tournament_size)
    best_idx = selected_idx[np.argmax(fitness_values[selected_idx])]
    return population[best_idx]

# Crossover
def crossover(parent1, parent2):
    if parent1.shape[0] > 1 and parent2.shape[0] > 1:
        crossover_point = random.randint(1, min(parent1.shape[0], parent2.shape[0]) - 1)
        child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
    return child1, child2

# Mutação
def mutate(individual, mutation_rate, grid_resolution):
    mutated = individual.copy()
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, individual.shape[0] - 1)
        mutated[mutation_point] = np.round(np.random.rand(1, 2) * 400 / grid_resolution) * grid_resolution
    if random.random() < mutation_rate:
        if random.random() < 0.5 and individual.shape[0] > 1:  # Remover PA
            mutation_point = random.randint(0, individual.shape[0] - 1)
            mutated = np.delete(mutated, mutation_point, axis=0)
        else:  # Adicionar PA
            new_pa = np.round(np.random.rand(1, 2) * 400 / grid_resolution) * grid_resolution
            mutated = np.vstack((mutated, new_pa))
    return mutated

# Loop de evolução
for gen in range(num_generations):
    fitness_values = np.array([evaluate_fitness(ind, clientes, max_distancia, max_capacidade_pa, max_num_pas, max_penalty, unattended_penalty)[0] for ind in population])
    
    new_population = []
    for _ in range(pop_size // 2):
        parent1 = tournament_selection(population, fitness_values)
        parent2 = tournament_selection(population, fitness_values)
        
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate, grid_resolution)
        child2 = mutate(child2, mutation_rate, grid_resolution)
        
        new_population.extend([child1, child2])
    
    population = new_population
    
    max_fitness = np.max(fitness_values)
    if max_fitness > best_fitness:
        best_fitness = max_fitness
        best_individual = population[np.argmax(fitness_values)]
    
    best_fitness_values.append(best_fitness)  # Armazenar o melhor fitness da geração atual
    print(f'Geração {gen + 1}: Melhor Fitness = {best_fitness}')

# Plotar o gráfico do melhor fitness a cada geração
plt.figure()
plt.plot(range(1, num_generations + 1), best_fitness_values, linewidth=2)
plt.title('Melhor Fitness por Geração')
plt.xlabel('Geração')
plt.ylabel('Melhor Fitness')
plt.grid(True)
plt.show()

# Visualizar a distribuição dos clientes e pontos de acesso
plt.figure()
plt.scatter(clientes['x'], clientes['y'], c='blue', label='Clientes')
if best_individual is not None:
    plt.scatter(best_individual[:, 0], best_individual[:, 1], c='red', marker='x', label='Pontos de Acesso')
plt.title('Distribuição dos Clientes e Pontos de Acesso')
plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.legend()
plt.grid(True)
plt.show()

# Salvar as coordenadas dos pontos de acesso encontrados
fitness, pa_largura_banda_usada = evaluate_fitness(best_individual, clientes, max_distancia, max_capacidade_pa, max_num_pas, max_penalty, unattended_penalty)
with open('melhorsolucao.txt', 'w') as file:
    for i in range(best_individual.shape[0]):
        porcentagem_uso = (pa_largura_banda_usada[i] / max_capacidade_pa) * 100
        file.write(f'{int(best_individual[i, 0])},{int(best_individual[i, 1])},{porcentagem_uso:.1f}%\n')

print('Arquivo melhorsolucao.txt gerado com sucesso!')
