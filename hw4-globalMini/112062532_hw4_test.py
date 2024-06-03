import numpy as np
import cma
from HomeworkFramework import Function

class RS_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)

            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            value = self.f.evaluate(func_num, solution)
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break            
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))
            
class PSO_optimizer(Function):
    def __init__(self, target_func, num_particles=30):
        super().__init__(target_func)
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func

        self.eval_times = 0

        self.num_particles = num_particles
        self.particles = np.random.uniform(self.lower, self.upper, (num_particles, self.dim))
        self.velocities = np.zeros((num_particles, self.dim))
        self.local_best_particles = np.copy(self.particles)
        self.local_best_values = np.full(num_particles, float('inf'))

        self.global_best_value = float('inf')
        self.global_best_particle = np.zeros(self.dim)

    def get_optimal(self):
        return self.global_best_particle, self.global_best_value

    def run(self, FES):
        w = 0.7
        c1, c2 = 1.2, 1.2

        while self.eval_times < FES:
            for i in range(self.num_particles):
                fitness = self.f.evaluate(self.target_func, self.particles[i])
                self.eval_times += 1
                if isinstance(fitness, str):  # Check if fitness is a string
                    if fitness == "ReachFunctionLimit":
                        print("Reached function limit; stopping optimization.")
                        return
                    else:
                        raise ValueError(f"Unexpected string return value from evaluate: {fitness}")
            
                if fitness < self.local_best_values[i]:
                    self.local_best_values[i] = fitness
                    self.local_best_particles[i] = np.copy(self.particles[i])

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_particle = np.copy(self.particles[i])

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = w * self.velocities[i] + c1 * r1 * (self.local_best_particles[i] - self.particles[i]) + c2 * r2 * (self.global_best_particle - self.particles[i])
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower, self.upper)
            print("optimal: {}\n".format(self.get_optimal()[1]))

class DE_optimizer(Function):
    def __init__(self, target_func, pop_size=50, mutation_factor=0.8, crossover_probability=0.7):
        super().__init__(target_func)
        
        # Boundaries, dimensions, and target function initialization from the framework
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        
        # Differential Evolution parameters
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability
        
        # Population initialization
        self.population = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        
        # Best solution initialization
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        
        # Evaluation counter initialization
        self.eval_times = 0
        
    def get_optimal(self):
        return self.optimal_solution, self.optimal_value
    
    def mutate(self, target_idx):
        # Choose three random indices different from target_idx
        idxs = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        
        # Mutant vector generation
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lower, self.upper)
        return mutant
    
    def crossover(self, target, mutant):
        # Create trial vector by mixing the target and mutant based on crossover_probability
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial = np.where(crossover_mask, mutant, target)
        return trial
    
    def select(self, target_idx, trial):
        # Evaluate the trial vector
        trial_fitness = self.f.evaluate(self.target_func, trial)
        self.eval_times += 1
        
        if trial_fitness == "ReachFunctionLimit":
            print("ReachFunctionLimit")
            return True
        
        # Selection process
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness
            
            # Update the best solution found
            if trial_fitness < self.optimal_value:
                self.optimal_solution = trial
                self.optimal_value = trial_fitness
                
        return False
    
    def run(self, FES):
        # Initial fitness evaluation for all individuals in the population
        for i in range(self.pop_size):
            self.fitness[i] = self.f.evaluate(self.target_func, self.population[i])
            self.eval_times += 1
            
            # Check if initial population provides a new best
            if self.fitness[i] < self.optimal_value:
                self.optimal_value = self.fitness[i]
                self.optimal_solution = self.population[i]
        
        # Evolution process
        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)
            
            for i in range(self.pop_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                if self.select(i, trial):
                    break  # Stop if we reach the function limit
            
            print("optimal: {}\n".format(self.get_optimal()[1]))


class CMAES_optimizer(Function):
    def __init__(self, target_func, seed=0, population_size=6, sigma=0.5):
        super().__init__(target_func)
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        # print("dimension", self.dim)
        self.eval_times = 0
        
        # CMA-ES parameters
        # self.population_size = self.dim * 4
        # self.population_size = self.dim * 5
        self.population_size = population_size
        # self.sigma = (self.upper-self.lower)/3  # Initial standard deviation
        self.sigma = sigma  # Initial standard deviation
        self.options = {
            'bounds': [self.lower, self.upper], 
            'popsize': self.population_size,   
            # 'seed': seed     
        }

        # Initialize the CMA-ES strategy
        self.es = cma.CMAEvolutionStrategy(self.dim * [0.5 * (self.lower + self.upper)], self.sigma, self.options)

        # Best solution tracking
        self.global_best_value = float('inf')
        self.global_best_particle = np.zeros(self.dim)

    def evaluate(self, x):
        return float(self.f.evaluate(2, x))

    def get_optimal(self):
        return self.global_best_particle, self.global_best_value, self.es.opts['seed']

    def run(self, FES):
        while not self.es.stop() and self.eval_times < FES:
            solutions = self.es.ask()  # Generate new sample solutions
            fitness = [self.f.evaluate(self.target_func, solution) for solution in solutions]
            self.eval_times += len(solutions)
            if "ReachFunctionLimit" in fitness:
                # print("Reached function limit; stopping optimization.")
                break

            self.es.tell(solutions, [f if isinstance(f, (float, int)) else float('inf') for f in fitness])  # Update the internal model

            valid_fitness = [f if isinstance(f, (float, int)) else float('inf') for f in fitness]
            if min(valid_fitness) < self.global_best_value:
                self.global_best_value = min(valid_fitness)
                self.global_best_particle = solutions[np.argmin(valid_fitness)]
            # print("Current Best Value: {}\n".format(self.global_best_value))

            if 'ftarget' in self.es.opts and self.global_best_value <= self.es.opts['ftarget']:
                print("Target function value reached.")
                break
            # print("optimal: {}\n".format(self.get_optimal()[1]))
        # print("seed: ", self.es.opts['seed'])

class CoDE_optimizer(Function):
    def __init__(self, target_func, pop_size=50, max_iters=100):
        super().__init__(target_func)
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.eval_times = 0

        
        # Initialize population
        self.population = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        self.fitness = np.array([self.f.evaluate(target_func, ind) for ind in self.population])
        
        self.global_best_value = np.min(self.fitness)
        self.global_best_particle = self.population[np.argmin(self.fitness)]
        
    def evaluate(self, x):
        return float(self.f.evaluate(2, x))
    
    def get_optimal(self):
        return self.global_best_particle, self.global_best_value
    
    def run(self, FES):
        trial_population = np.zeros((self.pop_size, self.dim))
        trial_fitness = np.full(self.pop_size, float('inf'))

        iters = 0
        while iters < self.max_iters and self.eval_times < FES:
            for i in range(self.pop_size):
                strategy = np.random.choice(['rand/1/bin', 'best/1/bin', 'current-to-best/1/bin'])
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = self.population[indices]
                # if strategy == 'rand/1/bin':
                # mutant = x1 + 0.5 * (x2 - x3)
                # elif strategy == 'best/1/bin':
                mutant = self.global_best_particle + 0.5 * (x2 - x3)
                # elif strategy == 'current-to-best/1/bin':
                    # mutant = self.population[i] + 0.5 * (self.global_best_particle - self.population[i]) + 0.5 * (x2 - x3)
                
                mutant = np.clip(mutant, self.lower, self.upper)
                fit_result = self.f.evaluate(self.target_func, mutant)
                self.eval_times += 1

                # Check if the evaluation result is a string indicating a limit
                if isinstance(fit_result, str):
                    if fit_result == "ReachFunctionLimit":
                        print("Reached function limit; stopping optimization.")
                        return True  # Optionally handle the limit in a specific way
                    continue  # Or handle other string cases as needed
                
                # Safely convert fitness result to float if necessary
                trial_fitness[i] = float(fit_result)
    
                if trial_fitness[i] < self.fitness[i]:
                    self.population[i] = mutant
                    self.fitness[i] = trial_fitness[i]
                    if trial_fitness[i] < self.global_best_value:
                        self.global_best_value = trial_fitness[i]
                        self.global_best_particle = mutant

                if self.eval_times >= FES:
                    break

            # print(f"Best Value at iteration {iters}: {self.global_best_value}")
            print("optimal: {}\n".format(self.get_optimal()[1]))
            iters += 1

class EDA_LS_optimizer(Function):
    def __init__(self, target_func, pop_size=50, max_iters=100):
        super().__init__(target_func)
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.target_func = target_func
        
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.eval_times = 0

        # Initialize population
        self.population = np.random.uniform(self.lower, self.upper, (self.pop_size, self.dim))
        self.fitness = np.array([self.f.evaluate(target_func, ind) for ind in self.population])
        
        self.global_best_value = np.min(self.fitness)
        self.global_best_particle = self.population[np.argmin(self.fitness)]
        
    def get_optimal(self):
        return self.global_best_particle, self.global_best_value
    
    def local_search(self, candidate):
        # Perform some local search strategy around candidate
        # For simplicity, perform a random walk
        step_size = 0.1 * (self.upper - self.lower)
        local_candidate = candidate + np.random.uniform(-step_size, step_size, self.dim)
        local_candidate = np.clip(local_candidate, self.lower, self.upper)
        local_fitness = self.f.evaluate(self.target_func, local_candidate)
        return local_candidate, local_fitness
    
    def run(self, FES):
        iters = 0
        while iters < self.max_iters and self.eval_times < FES:
            # Estimate the distribution of the best solutions
            mean = np.mean(self.population, axis=0)
            std_dev = np.std(self.population, axis=0) + 1e-6  # avoid division by zero

            # Generate new solutions by sampling the estimated distribution
            self.population = np.random.normal(mean, std_dev, (self.pop_size, self.dim))
            self.population = np.clip(self.population, self.lower, self.upper)

            # Evaluate new solutions
            # self.fitness = np.array([self.f.evaluate(self.target_func, ind) for ind in self.population])
            # self.eval_times += len(self.population)
            for i in range(self.pop_size):
                fit_result = self.f.evaluate(self.target_func, self.population[i])
                self.eval_times += 1

                # Check if fit_result is a string indicating an error
                if isinstance(fit_result, str):
                    if fit_result == "ReachFunctionLimit":
                        print("Reached function limit; stopping optimization.")
                        return True  # Optionally handle the limit in a specific way
                    continue  # Skip further processing for this individual

                self.fitness[i] = float(fit_result)  # Safely convert to float

            # Apply local search on the best found solution
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.global_best_value:
                self.global_best_value = self.fitness[best_idx]
                self.global_best_particle = self.population[best_idx]
            
            # Local search
            self.global_best_particle, new_fitness = self.local_search(self.global_best_particle)
            if new_fitness < self.global_best_value:
                self.global_best_value = new_fitness
            
            print(f"Best Value at iteration {iters}: {self.global_best_value}")
            iters += 1
        
        return self.get_optimal()

if __name__ == '__main__':
    func_num = 1
    fes = 0
    while(func_num<=4):
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500
        time = 0
        min_val = 1000
        min_input = None
        best_seed = 0
        op = None
        #function1: 1000, function2: 1500, function3: 2000, function4: 2500
        while time<50000:
            # you should implement your optimizer
            op = CMAES_optimizer(func_num)
            op.run(fes)
            
            best_input, best_value, seed = op.get_optimal()
            # print(best_input, best_value)
            if best_value < min_val:
                print("update papram")
                min_val = best_value
                min_input = best_input
                best_seed = seed
            time+=1
            print(time)
            # print("---------------------------------------------------------------------------")
            # print()

            # change the name of this file to your student_ID and it will output properlly
        with open("410885034_test_pop_6_function{}.txt".format(func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(min_input[i]))
            f.write("{}\n".format(min_val))
            f.write("{}\n".format(best_seed))
        func_num += 1 
    
