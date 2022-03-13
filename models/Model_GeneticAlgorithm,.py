import Libraries
import functools
import random
import sys
import time
import typing as t

class NatrualSelectionGA:
    
    def __init__ (
            self,
            population_size: int,
            genes_alphabet: str,
            chromosome_length: int,
            fitness_function: t.Callable[[str], [float]],
            gene_chance_of_mutation: int,
            max_stale_generations: int,
            verbose: bool = False
            ) -> None:
                self.population_size = population_size
                self.genes_alphabet = genes_alphabet
                self.chromosome_length = chromosome_length
                self.gene_chance_of_mutation = gene_chance_of_mutation
                self.max_stale_generations = max_stale_generations
                self.verbose = verbose
        
                # Let's memoize the fitness function for improved performance.
                self.fitness_function: t.Callable[[str], float] = (functools.lru_cache(maxsize=131072))(fitness_function)
                
    def run(self) -> str:
        population = self.gen_initial_population()
        generaion_number = 1
        if self.verbose and self.population_size <= 50:
            print (f"Initial population : {population}")
        
        # Best individual found so far
        best_indv = self.get_fittest_individual (population)
        
        # Score of the best individual
        best_score = self.finess_function(best_indv)
        
        # Number of the generation where the best individual appeared
        best_generation = generation_number
        
        generations_since_best = 0
        
        while generations_since_best < self.max_stale_generations:
            population = self.gen_new_generation (population)
            generation_number += 1
            
            generation_fittest = self.get_fittest_individual(population)
            generation_fittest_score = self.fitness_function(generation_fittest)
            
            if generation_fittest_score > best_score:
                best_score = generation_fittest_score
                best_indv = generation_fittest
                best_generation = generaion_number
                generations_since_best = 0
                
                if self.verbose:
                    override_line(
                        f"[Generation {generation_number:4}] "
                        f"Fittest chromosome: {generation_fittest} "
                        f"(score {generation_fittest_score:10})\n"
                        )
            else:
                generations_since_best += 1
                
                if self.verbose:
                    override_line(
                        f"Generation {generation_number}: "
                        f"Fittest: {generation_fittest} "
                        f"(score {generation_fittest_score} "
                        f"elapsed time {(time.time() - start_time):2.2f}s)"
                    )
        
        if self.verbose:
            total_time = time.time() - start_time
            override_line(
            f"Fittest genome: {best_indv} "
            f"(generation {best_generation}, "
            f"score: {best_score})\n"
        )
            print(f"Generations: {generation_number}")
            print(
                f"Elapsed time: {total_time:.2f}s "
                f"(avg {total_time / generation_number:.2}s)"
            )

        return best_indv
                
        
    def gen_initial_population(self) -> t.List[str]:
        return [self.gen_random_chromosome() for _ in range(self.population_size)]
    
    def gen_random_chromosome(self) -> str:
        return "".join(random.choices(self.genes_alphabet, k=self.chromosome_length))
    
    def gen_new_generation (self, old_generation: t.List[str]) -> t.List[str]:
        population_fitness = self.compute_population_fitness(old_generation)
        
        fit_individuals_iter = iter (
            self.sample_individuals(
                population = old_generation,
                weights = population_fitness,
                sample_size = 2 * self.population_size,
                )
            )
        
        new_generation = [self.mate(fit_individual, next(fit_individuals_iter)) for fit)individual in fit_individuals_iter]
        return new_generation
    
    
        
        
                                                                                                                            
            
                                        	