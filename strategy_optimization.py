import numpy as np
import random
from scipy.optimize import differential_evolution

# 遗传算法优化策略
def optimize_strategy(parameters, historical_data, generations=100, population_size=50, mutation_rate=0.1, method="genetic"):
    if method == "genetic":
        return optimize_with_ga(parameters, historical_data, generations, population_size, mutation_rate)
    elif method == "pso":
        return optimize_with_pso(parameters, historical_data)
    elif method == "grid_search":
        return grid_search(parameters, historical_data)
    else:
        raise ValueError("未知优化方法: {}".format(method))

# 遗传算法
def optimize_with_ga(parameters, historical_data, generations, population_size, mutation_rate):
    population = initialize_population(parameters, population_size)
    
    for generation in range(generations):
        fitness_scores = evaluate_population(population, historical_data)
        population = evolve_population(population, fitness_scores, mutation_rate)
    
    best_params = get_best_solution(population, fitness_scores)
    return best_params

# 粒子群优化（PSO）
def optimize_with_pso(parameters, historical_data):
    bounds = [(v[0], v[1]) for v in parameters.values()]

    def pso_fitness(params):
        param_dict = {key: params[i] for i, key in enumerate(parameters.keys())}
        return -evaluate_strategy(param_dict, historical_data)  # 最小化目标函数

    result = differential_evolution(pso_fitness, bounds, maxiter=100)
    best_params = {key: result.x[i] for i, key in enumerate(parameters.keys())}
    return best_params

# 网格搜索
def grid_search(parameters, historical_data):
    keys, values = zip(*parameters.items())
    best_score = -np.inf
    best_params = None

    for combination in product(*(np.linspace(v[0], v[1], 10) for v in values)):  # 网格10步
        param_dict = dict(zip(keys, combination))
        score = evaluate_strategy(param_dict, historical_data)
        if score > best_score:
            best_score = score
            best_params = param_dict

    return best_params

# 初始化种群
def initialize_population(parameters, population_size):
    return [randomize_parameters(parameters) for _ in range(population_size)]

# 随机初始化参数
def randomize_parameters(parameters):
    return {key: random.uniform(value[0], value[1]) for key, value in parameters.items()}

# 评估种群
def evaluate_population(population, historical_data):
    return [evaluate_strategy(params, historical_data) for params in population]

# 评估策略
def evaluate_strategy(params, historical_data):
    # 示例：可以替换为策略的回测逻辑，返回策略收益或评估指标
    return np.random.rand()  # 随机返回适应度值

# 进化种群
def evolve_population(population, fitness_scores, mutation_rate):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
    next_generation = sorted_population[:2]  # 保留最优解
    
    while len(next_generation) < len(population):
        parent1, parent2 = random.sample(sorted_population[:10], 2)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            child = mutate(child)
        next_generation.append(child)
    
    return next_generation

# 基因交叉
def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = (parent1[key] + parent2[key]) / 2
    return child

# 基因突变
def mutate(parameters):
    key = random.choice(list(parameters.keys()))
    parameters[key] += random.uniform(-0.1, 0.1)  # 随机变动
    return parameters

# 获取最佳解决方案
def get_best_solution(population, fitness_scores):
    best_idx = np.argmax(fitness_scores)
    return population[best_idx]

# 动态调整策略（示例）
def dynamic_adjustment(volatility, base_amount, adjustment_factor=1.5):
    adjusted_amount = base_amount / (1 + adjustment_factor * volatility)
    return adjusted_amount
