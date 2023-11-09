from mido import MidiFile, MidiTrack, MetaMessage, Message
import math
import music21

filename = 'input1.mid'
mid = MidiFile(filename)

# find the number of beats (the number of required chords)
ticks = 0
for msg in mid.tracks[1]:
    if msg.is_meta or msg.type == 'program_change':
        continue
    ticks = ticks + msg.time
beats = math.ceil(ticks / mid.ticks_per_beat)

# add notes played simultaneously with each chord
notes = [None] * beats
ticks = 0
for msg in mid.tracks[1]:
    if msg.is_meta or msg.type == 'program_change':
        continue
    ticks = ticks + msg.time
    if msg.type == 'note_off':
        continue
    else:
        if ticks % mid.ticks_per_beat == 0:
            notes[int(ticks / mid.ticks_per_beat)] = msg.note

# find the key
score = music21.converter.parse(filename)
key = score.analyze('key')
key = key.tonic.name + key.mode

# chords of the keys that occurred in the input file
chords = {'Dminor': [[38, 41, 45], [40, 43, 46], [41, 45, 48], [43, 46, 50], [45, 48, 52], [46, 50, 53], [48, 52, 55]],
          'Fmajor': [[41, 45, 48], [43, 46, 50], [45, 48, 52], [46, 50, 53], [48, 52, 55], [38, 41, 45], [40, 43, 46]],
          'Eminor': [[40, 43, 47], [42, 45, 48], [43, 47, 50], [45, 48, 52], [47, 50, 54], [48, 52, 55], [50, 54, 57]]}

import random

# represents every single individual in the population
class Individual:

    def __init__(self, genotype: str, fitness: int):
        self.genotype = genotype
        self.fitness = fitness

    def __repr__(self):
        return "Individual/genotype = " + self.genotype + " Fitness = " + str(self.fitness)

# a class that evaluates how good the individual is
class FitnessEvaluator:

    def __init__(self):
        pass

    # a function that depends on if the chord matches the notes that were played with it simultaneously
    def evaluate(self, genotype: str):
        fitness = 0
        for i in range(len(genotype)):
            if self.isDissonant(genotype[i], i):
                fitness = fitness - 1
            if self.isRoot(genotype[i], i):
                fitness = fitness + 2
            if i != len(genotype)-1:
                if self.areClose(genotype[i], genotype[i + 1]):
                    fitness = fitness + 1
        return fitness

    # uses the dissonance intervals to see if any of the chord notes sounds dissonant with the current note
    def isDissonant(self, chord_id, index):
        chord = chords[key][ord(chord_id) - 48]
        note = notes[index]
        if note is None:
            return False
        for chord_note in chord:
            if math.fabs(chord_note % 12 - note % 12) in [1, 2, 6, 10, 11]:
                return True
        return False

    # a function that checks if the root of the played chord is the same the note currently played
    def isRoot(self, chord_id, index):
        chord = chords[key][ord(chord_id) - 48]
        chord_root = chord[0]
        note = notes[index]
        if note is None:
            return False
        if chord_root % 12 == note % 12:
            return True
        return False

    # a function that determines if the consecutive chords are "close" or easily played
    def areClose(self, chord_id1, chord_id2):
        chord1 = chords[key][ord(chord_id1) - 48]
        chord2 = chords[key][ord(chord_id2) - 48]
        dist = 0
        for i in range(len(chord1)):
            dist += math.fabs(chord1[i] - chord2[i])
        if dist > 6:
            return False
        else:
            return True

# a factory class of the class individual
class IndividualFactory:

    def __init__(self, genotype_length: int, fitness_evaluator: FitnessEvaluator):
        self.genotype_length = genotype_length
        self.fitness_evaluator = fitness_evaluator

    def with_random_genotype(self):
        random_genotype = ''
        for _ in range(self.genotype_length):
            random_genotype += chr(random.randint(0, 6) + 48)
        fitness = self.fitness_evaluator.evaluate(random_genotype)
        return Individual(random_genotype, fitness)

    def with_set_genotype(self, genotype: str):
        fitness = self.fitness_evaluator.evaluate(genotype)
        return Individual(genotype, fitness)


# a class that manages the population of individuals
class Population:

    def __init__(self, individuals):
        self.individuals = individuals

    def get_the_fittest(self, n: int):
        self.__sort_by_fitness()
        return self.individuals[:n]

    def __sort_by_fitness(self):
        self.individuals.sort(key=self.__individual_fitness_sort_key, reverse=True)

    def __individual_fitness_sort_key(self, individual: Individual):
        return individual.fitness

# the factory class of the population class
class PopulationFactory:

    def __init__(self, individual_factory: IndividualFactory):
        self.individual_factory = individual_factory

    def with_random_individuals(self, size: int):
        individuals = []
        for i in range(size):
            individuals.append(self.individual_factory.with_random_genotype())
        return Population(individuals)

    def with_individuals(self, individuals):
        return Population(individuals)

# a class to choose the parents to reproduce the next generation
class ParentSelector:

    def select_parents(self, population: Population):
        total_fitness = 0
        fitness_scale = []
        for index, individual in enumerate(population.individuals):
            total_fitness += individual.fitness
            if index == 0:
                fitness_scale.append(individual.fitness)
            else:
                fitness_scale.append(individual.fitness + fitness_scale[index - 1])

        # Store the selected parents
        mating_pool = []
        # Equal to the size of the population
        number_of_parents = len(population.individuals)
        # How fast we move along the fitness scale
        fitness_step = total_fitness / number_of_parents
        random_offset = random.uniform(0, fitness_step)

        # Iterate over the parents size range and for each:
        # - generate pointer position on the fitness scale
        # - pick the parent who corresponds to the current pointer position and add them to the mating pool
        current_fitess_pointer = random_offset
        last_fitness_scale_position = 0
        for index in range(len(population.individuals)):
            for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
                if fitness_scale[fitness_scale_position] >= current_fitess_pointer:
                    mating_pool.append(population.individuals[fitness_scale_position])
                    last_fitness_scale_position = fitness_scale_position
                    break
            current_fitess_pointer += fitness_step

        return mating_pool

# a class to implement the single point crossover
class SinglePointCrossover:

    def __init__(self, individual_factory: IndividualFactory):
        self.individual_factory = individual_factory

    def crossover(self, parent_1: Individual, parent_2: Individual):
        crossover_point = random.randint(0, len(parent_1.genotype))
        genotype_1 = self.__new_genotype(crossover_point, parent_1, parent_2)
        genotype_2 = self.__new_genotype(crossover_point, parent_2, parent_1)
        child_1 = self.individual_factory.with_set_genotype(genotype=genotype_1)
        child_2 = self.individual_factory.with_set_genotype(genotype=genotype_2)
        return child_1, child_2

    def __new_genotype(self, crossover_point: int, parent_1: Individual, parent_2: Individual):
        return parent_1.genotype[:crossover_point] + parent_2.genotype[crossover_point:]

# the mutator class
class Mutator:

    def __init__(self, individual_factory: IndividualFactory):
        self.individual_factory = individual_factory

    def mutate(self, individual: Individual):
        mutated_genotype = list(individual.genotype)
        mutation_probability = 1 / len(individual.genotype)
        for index, gene in enumerate(individual.genotype):
            if random.random() < mutation_probability:
                new_value = random.randint(0, 6)
                mutated_genotype[index] = chr(new_value + 48)
        return self.individual_factory.with_set_genotype(genotype="".join(mutated_genotype))

# the class that generates the next generation
class Breeder:

    def __init__(self,
                 single_point_crossover: SinglePointCrossover,
                 mutator: Mutator):
        self.single_point_crossover = single_point_crossover
        self.mutator = mutator

    def produce_offspring(self, parents):
        offspring = []
        number_of_parents = len(parents)
        for index in range(int(number_of_parents / 2)):
            parent_1, parent_2 = self.__pick_random_parents(parents, number_of_parents)
            child_1, child_2 = self.single_point_crossover.crossover(parent_1, parent_2)
            child_1_mutated = mutator.mutate(child_1)
            child_2_mutated = mutator.mutate(child_2)
            offspring.extend((child_1_mutated, child_2_mutated))

        return offspring

    def __pick_random_parents(self, parents, number_of_parents: int):
        parent_1 = parents[random.randint(0, number_of_parents - 1)]
        parent_2 = parents[random.randint(0, number_of_parents - 1)]
        return parent_1, parent_2

# the environment that contains the whole process of evolution
class Environment:

    def __init__(self,
                 population_size: int,
                 parent_selector: ParentSelector,
                 population_factory: PopulationFactory,
                 breeder: Breeder):
        self.population_factory = population_factory
        self.population = self.population_factory.with_random_individuals(size=population_size)
        self.parent_selector = parent_selector
        self.breeder = breeder

    def update(self):
        parents = self.parent_selector.select_parents(self.population)
        next_generation = breeder.produce_offspring(parents)
        self.population = population_factory.with_individuals(next_generation)

    def get_the_fittest(self, n: int):
        return self.population.get_the_fittest(n)

# a function that creates a track of chords to be added to the original track
def generate_track(genome: str, tpb: int, beats: int) -> MidiTrack:
    track = MidiTrack()
    meta = MetaMessage('track_name', name='Elec. Piano (Classic)', time=0)
    first_message = Message('program_change', channel=0, program=0, time=0)
    track.append(meta)
    track.append(first_message)

    for i in range(beats):
        chord = chords[key][ord(genome[i]) - 48]
        for j in chord:
            track.append(Message('note_on', note=j, time=0)) # play at the same time
        for j in range(len(chord)):
            offset = 0
            if j == 0:
                offset = tpb # make the delta time a single beat
            track.append(Message('note_off', note=chord[j], time=offset))

    return track


TOTAL_GENERATIONS = 3000
POPULATION_SIZE = 100
GENOTYPE_LENGTH = beats

current_generation = 1

fitness_evaluator = FitnessEvaluator()
individual_factory = IndividualFactory(GENOTYPE_LENGTH, fitness_evaluator)
population_factory = PopulationFactory(individual_factory)
single_point_crossover = SinglePointCrossover(individual_factory)
mutator = Mutator(individual_factory)
breeder = Breeder(single_point_crossover, mutator)
parent_selector = ParentSelector()
environment = Environment(POPULATION_SIZE,
                          parent_selector,
                          population_factory,
                          breeder)

highest_fitness_list = []
while current_generation <= TOTAL_GENERATIONS:
    print(current_generation)
    fittest = environment.get_the_fittest(1)[0]
    highest_fitness_list.append(fittest.fitness)
    environment.update()
    current_generation += 1

print("Stopped at generation " + str(current_generation - 1) + ". The fittest individual: ")
print(fittest)
track = generate_track(fittest.genotype, mid.ticks_per_beat, beats)
mid.tracks.append(track)
mid.save('output1.mid')