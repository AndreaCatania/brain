#pragma once

#include "brain/NEAT/neat_genome.h"
#include <random>

namespace brain {

class NtSpecies;
class NtOrganism;

/**
 * @brief The NtEpochStatistics struct is used to track the changes that the
 * population doesn during a specific epoch
 */
struct NtEpochStatistics {

	operator std::string() const {

		return "\n{\"epoch\":" + itos(epoch) + "," +
			   "\n\"is_epoch_advanced\":" + (is_epoch_advanced ? "true" : "false") + "," +
			   "\n\"pop_champion_fitness\":" + rtos(pop_champion_fitness) + "," +
			   "\n\"pop_champion_species_id\":" + itos(pop_champion_species_id) + "," +
			   "\n\"species_count\":" + itos(species_count) + "," +
			   "\n\"species_young_count\":" + itos(species_young_count) + "," +
			   "\n\"species_stagnant_count\":" + itos(species_stagnant_count) + "," +
			   "\n\"species_avg_ages\":" + itos(species_avg_ages) + "," +
			   "\n\"species_best_id\":" + itos(species_best_id) + "," +
			   "\n\"species_best_age\":" + itos(species_best_age) + "," +
			   "\n\"species_best_offspring_pre_steal\":" + itos(species_best_offspring_pre_steal) + "," +
			   "\n\"species_best_offspring\":" + itos(species_best_offspring) + "," +
			   "\n\"species_best_champion_offspring\":" + itos(species_best_champion_offspring) + "," +
			   "\n\"species_best_is_died\":" + (species_best_is_died ? "true" : "false") + "," +
			   "\n\"pop_avg_fitness\":" + rtos(pop_avg_fitness) + "," +
			   "\n\"pop_is_stagnant\":" + (pop_is_stagnant ? "true" : "false") + "," +
			   "\n\"pop_epoch_last_improvement\":" + itos(pop_epoch_last_improvement) + "," +
			   "\n\"pop_stolen_cribs\":" + itos(pop_stolen_cribs) + "," +
			   "\n\"reproduction_champion_mutate_weights\":" + rtos(reproduction_champion_mutate_weights) + "," +
			   "\n\"reproduction_champion_add_random_link\":" + rtos(reproduction_champion_add_random_link) + "," +
			   "\n\"reproduction_mate_multipoint\":" + rtos(reproduction_mate_multipoint) + "," +
			   "\n\"reproduction_mate_multipoint_avg\":" + rtos(reproduction_mate_multipoint_avg) + "," +
			   "\n\"reproduction_mate_singlepoint\":" + rtos(reproduction_mate_singlepoint) + "," +
			   "\n\"reproduction_mutate_add_random_link\":" + rtos(reproduction_mutate_add_random_link) + "," +
			   "\n\"reproduction_mutate_add_random_neuron\":" + rtos(reproduction_mutate_add_random_neuron) + "," +
			   "\n\"reproduction_mutate_weights\":" + rtos(reproduction_mutate_weights) + "," +
			   "\n\"reproduction_mutate_toggle_link_activation\":" + rtos(reproduction_mutate_toggle_link_activation) + "}";
	}

	void clear() {
		epoch = 0;
		is_epoch_advanced = false;
		pop_champion_fitness = 0.f;
		pop_champion_species_id = -1;
		species_count = 0;
		species_young_count = 0;
		species_stagnant_count = 0;
		species_avg_ages = 0;
		species_best_id = 0;
		species_best_age = 0;
		species_best_offspring_pre_steal = 0;
		species_best_offspring = 0;
		species_best_champion_offspring = 0;
		species_best_is_died = false;
		pop_avg_fitness = 0;
		pop_is_stagnant = false;
		pop_epoch_last_improvement = 0;
		pop_stolen_cribs = 0;
		reproduction_champion_mutate_weights = 0;
		reproduction_champion_add_random_link = 0;
		reproduction_mate_multipoint = 0;
		reproduction_mate_multipoint_avg = 0;
		reproduction_mate_singlepoint = 0;
		reproduction_mutate_add_random_link = 0;
		reproduction_mutate_add_random_neuron = 0;
		reproduction_mutate_weights = 0;
		reproduction_mutate_toggle_link_activation = 0;
	}

	/**
	 * @brief The epoch when this statistic was recordered
	 */
	uint32_t epoch;

	/**
	 * @brief is_epoch_advanced
	 */
	bool is_epoch_advanced;

	/**
	 * @brief Population champion personal fitness
	 */
	real_t pop_champion_fitness;

	/**
	 * @brief The species ID of the pop champion
	 */
	int pop_champion_species_id;

	/**
	 * @brief species_count
	 */
	int species_count;

	/**
	 * @brief species_young_count
	 */
	int species_young_count;

	/**
	 * @brief species_stagnant_count
	 */
	int species_stagnant_count;

	/**
	 * @brief species_avg_ages
	 */
	int species_avg_ages;

	/**
	 * @brief  The best species id
	 */
	uint32_t species_best_id;

	/**
	 * @brief The best species age
	 */
	uint32_t species_best_age;

	/**
	 * @brief species_best_offspring_pre_steal
	 */
	int species_best_offspring_pre_steal;

	/**
	 * @brief species_best_offspring
	 */
	int species_best_offspring;

	/**
	 * @brief species_best_champion_offspring
	 */
	int species_best_champion_offspring;

	/**
	 * @brief species_best_is_died
	 */
	bool species_best_is_died;

	/**
	 * @brief The organisms avg fitness (NOTE: Not personal fitness)
	 */
	real_t pop_avg_fitness;

	/**
	 * @brief pop_is_stagnant
	 */
	bool pop_is_stagnant;

	/**
	 * @brief pop_epoch_last_improvement
	 */
	uint32_t pop_epoch_last_improvement;

	/**
	 * @brief pop_stolen_cribs
	 */
	int pop_stolen_cribs;

	/**
	 * @brief reproduction_champion_mutate_weights
	 */
	int reproduction_champion_mutate_weights;

	/**
	 * @brief reproduction_champion_add_random_link
	 */
	int reproduction_champion_add_random_link;

	/**
	 * @brief reproduction_mate_multipoint
	 */
	int reproduction_mate_multipoint;

	/**
	 * @brief reproduction_mate_multipoint_avg
	 */
	int reproduction_mate_multipoint_avg;

	/**
	 * @brief reproduction_mate_singlepoint
	 */
	int reproduction_mate_singlepoint;

	/**
	 * @brief reproduction_mutate_add_random_link
	 */
	int reproduction_mutate_add_random_link;

	/**
	 * @brief reproduction_mutate_add_random_neuron
	 */
	int reproduction_mutate_add_random_neuron;

	/**
	 * @brief reproduction_mutate_weights
	 */
	int reproduction_mutate_weights;

	/**
	 * @brief reproduction_mutate_toggle_link_activation
	 */
	int reproduction_mutate_toggle_link_activation;
};

/**
 * @brief The NtPopulationSettings struct is a utility structure used to initialize
 * easily the population settings.
 */
struct NtPopulationSettings {

	/**
	 * @brief seed this is used to change the behaviour of the population
	 */
	uint64_t seed = 1;

	/**
	 * @brief learning_deviation is a parameter used to define the range of tests
	 * that is possible to do during weight adjusting.
	 *
	 * Note: a too small value could make the genome finding too difficult, and
	 * a possible good genome could die in the process
	 *
	 * IMPORTANT: This parameter is really important for the result
	 */
	real_t learning_deviation = 3.f;

	/**
	 * @brief genetic_compatibility_threshold is used to determine the species
	 * organisms, during genetic comparison.
	 *
	 * Lowering this mean have more species and thus be more selective.
	 *
	 * A too huge threshold can make the niching process to lose some champions
	 * prematurely since not yet fully formed.
	 * Opposite a too small threshold can make the niching process to take
	 * too much champion with more or less same traits.
	 *
	 * So a too huge or too small threshold can make the niching process to fail.
	 */
	real_t genetic_compatibility_threshold = 3.f;

	/**
	 * @brief genetic_disjoints_significance is used during the genetic comparison
	 * between two genomes.
	 *
	 * Changing this parameter changes the significance of the disjoints during
	 * genome comparison and will impact on the species size and differentiation
	 *
	 * The provided defaults parameters (disjoint 1, excess 1, weights 0.4)
	 * make sure to give more importance to the network topology rather
	 * the weights. So in the same species we will find more networks with the
	 * same topology but different weights
	 *
	 * N.B. Checks the Genetic::compatibility function doc to understand what
	 * Disjoint, Excess, Weight mean.
	 */
	real_t genetic_disjoints_significance = 1.f;

	/**
	 * @brief genetic_excesses_significance is used during the genetic comparison
	 * between two genomes.
	 *
	 * Changing this parameter changes the significance of the excess during
	 * genome comparison and will impact on the species size and differentiation
	 *
	 * The provided defaults parameters (disjoint 1, excess 1, weights 0.4)
	 * make sure to give more importance to the network topology rather
	 * the weights. So in the same species we will find more networks with the
	 * same topology but different weights
	 *
	 * N.B. Checks the Genetic::compatibility function doc to understand what
	 * Disjoint, Excess, Weight mean.
	 */
	real_t genetic_excesses_significance = 1.f;

	/**
	 * @brief genetic_weights_significance is used during the genetic comparison
	 * between two genomes.
	 *
	 * Changing this parameter changes the significance of the weights during
	 * genome comparison and will impact on the species size and differentiation
	 *
	 * The provided defaults parameters (disjoint 1, excess 1, weights 0.4)
	 * make sure to give more importance to the network topology rather
	 * the weights. So in the same species we will find more networks with the
	 * same topology but different weights
	 *
	 * N.B. Checks the Genetic::compatibility function doc to understand what
	 * Disjoint, Excess, Weight mean.
	 */
	real_t genetic_weights_significance = 0.4f;

	/**
	 * @brief genetic_mate_prob is used to define the probability for
	 * a genome to mate with another one, instead to mutate.
	 */
	real_t genetic_mate_prob = 0.3f;

	/**
	 * @brief genetic_mate_inside_species_threshold is used to controll where the
	 * other parent (dad) should be taken within the same species of the mom or
	 * outside.
	 */
	real_t genetic_mate_inside_species_threshold = 0.8f;

	/**
	 * @brief genetic_mating_* are used to decide what type of crossover operation
	 * should occur more often.
	 *
	 * These parameters will be normalized (If all are the same they will have
	 * the same probability)
	 */
	real_t genetic_mate_multipoint_threshold = 0.5;
	real_t genetic_mate_multipoint_avg_threshold = 0.5;
	real_t genetic_mate_singlepoint_threshold = 0.; // IMPORTANT: Doesn't work so well

	/**
	 * @brief genetic_mutate_* are used to decide what type of mutation should
	 * occur more often.
	 *
	 * These parameters will be normalized (If all are the same they will have
	 * the same probability)
	 */
	real_t genetic_mutate_add_link_porb = 0.1f;
	real_t genetic_mutate_add_node_prob = 0.05f;
	real_t genetic_mutate_link_weight_prob = 0.8f;
	real_t genetic_mutate_link_weight_uniform_prob = 0.9f; // 10% of use Cold gaussian
	real_t genetic_mutate_toggle_link_enable_prob = 0.05f;

	/**
	 * @brief genetic_spawn_recurrent_link_threshold goes from 0 to 1 and is used
	 * to determinates the probability to spawn a recurrent link instead of a
	 * regular one.
	 *
	 * The reccurrent links are links that move the data in the opposite direction
	 * (Right to Left), due to this characteristics these links returns the value
	 * of the previous iterations.
	 * This means that these links are useful to remember the previous datas and
	 * thus the taken decision is affected by the previous datas.
	 */
	real_t genetic_mutate_add_link_recurrent_prob = 0.05f;

	/**
	 * @brief species_youngness_age_threshold the ages within a species is considered
	 * young and so is protected
	 */
	int species_youngness_age_threshold = 10;

	/**
	 * @brief species_youngness_multiplier the fitness multiplier applied to the
	 * young species
	 */
	real_t species_youngness_multiplier = 2.f;

	/**
	 * @brief species_stagnant_age_threshold the ages without improvements
	 * to consider a species stagnant
	 */
	int species_stagnant_age_threshold = 15;

	/**
	 * @brief species_stagnant_multiplier the penalty multiplier applied to the
	 * stagnant species
	 */
	real_t species_stagnant_multiplier = 0.01f;

	/**
	 * @brief species_survival_ratio is used to know the percetage of survival
	 * inside the population.
	 * This must be always between 0 and 1.
	 */
	real_t species_survival_ratio = 0.5f;

	/**
	 * @brief population_cribs_stealing is used to define how much cribs should be
	 * stealed from the worst species.
	 *
	 * These cribs are assigned in this order:
	 *	1/5 are given to the two not dying best species
	 *	1/10 are give to the third not dying best species
	 *	The rest is assigned to all other not dying species randomly
	 *	If remains yet cribs to allocate, will be given them all to the best species
	 *
	 * With these extra cribs the species will have the possibility to have more
	 * offsprings, in this way is possible to promote the growing of the best
	 * species.
	 *
	 * Also the assigned cribs will be used to create some clones of the species
	 * champion, where the most part will have only the weight mutated
	 */
	int cribs_stealing = 20;

	/**
	 * @brief cribs_stealing_limit is used to stop the stealing when they get
	 * below this value.
	 * In thi way is possible to protect the specie from the death due to the
	 * stealing.
	 */
	int cribs_stealing_limit = 2;

	/**
	 * @brief species_stealing_protection_age_threshold is used to protect the
	 * species from stealing even if they perform low until they pass this threshold
	 */
	int cribs_stealing_protection_age_threshold = 3;

	/**
	 * @brief population_stagnant_age_thresold is used to detect if the population
	 * become stagnant and is not able to improve more.
	 */
	int population_stagnant_age_thresold = 15;
};

/**
 * @brief The NtPopulation class is responsible for the born, grow, death of
 * each member.
 *
 * This class is the API of the NEAT.
 */
class NtPopulation {

	friend class NtSpecies;

	/**
	 * @brief The population size
	 */
	const int population_size;

	/**
	 * @brief Customizable settings of the population
	 */
	NtPopulationSettings settings;

	/**
	 * @brief innovation_number is an incremental number that is used to
	 * mark and so track all changes to the organism genome.
	 *
	 * This is the last innovation number assigned
	 */
	uint32_t innovation_number;

	/**
	 * @brief species_last_index used to give a unique ID to the species
	 */
	uint32_t species_last_index;

	/**
	 * @brief rand_generator is used to generate a random number
	 */
	std::default_random_engine rand_generator;

	/**
	 * @brief gaussian_distribution is used to generate a random number within
	 * the gaussina distribution
	 */
	std::normal_distribution<real_t> gaussian_distribution;

	/**
	 * @brief species is the array of species of this population
	 */
	std::vector<NtSpecies *> species;

	/**
	 * @brief organisms are the list of all organism of this population.
	 * This class is fully responsible for the memory management
	 */
	std::vector<NtOrganism *> organisms;

	/**
	 * @brief epoch counter
	 */
	uint32_t epoch;

	/**
	 * @brief best_personal_fitness is the best fitness of the population champion
	 * ever.
	 * This is used to understand if the population is stagnant.
	 */
	real_t best_personal_fitness;

	/**
	 * @brief epoch_last_improvement is used to track the last improvements over
	 * the epochs and so understand if the population is stagnant.
	 */
	int epoch_last_improvement;

	/**
	 * @brief innovations list of innovations
	 */
	std::vector<NtInnovation> innovations;

	/**
	 * @brief champion_genome This is the champion genome of the past epoch.
	 */
	NtGenome champion_genome;

	/**
	 * @brief statistic Used to track what the epoch does
	 */
	NtEpochStatistics statistics;

public:
	/**
	 * @brief NtPopulation construct the population by spawning each member
	 * with the same traits of the ancestor_genome.
	 *
	 * (This mean that the population will have the same structure of the ancestor
	 * with only link weight mutated)
	 *
	 * Then the spawned organism will be speciated
	 *
	 * @param p_ancestor_genome
	 * @param p_population_size
	 * @param p_settings
	 */
	NtPopulation(
			const NtGenome &p_ancestor_genome,
			int p_population_size,
			NtPopulationSettings &p_settings);

	/**
	 * @brief NtPopulation Destructor
	 */
	~NtPopulation();

	/**
	 * @brief get_epoch returns the current population epoch
	 * @return
	 */
	uint32_t get_epoch() const;

	/**
	 * @brief returns the population size
	 * @return
	 */
	uint32_t get_population_size() const;

	/**
	 * @brief organism_get_network returns the neural network on this organism
	 * @param p_organism_i
	 * @return
	 */
	const SharpBrainArea *organism_get_network(uint32_t p_organism_i) const;

	/**
	 * @brief organism_set_fitness is used to tell how this organism is doing.
	 * Higher mean better
	 * @param p_organism_i
	 * @param p_fitness
	 */
	void organism_set_fitness(
			uint32_t p_organism_i,
			real_t p_fitness);

	/**
	 * @brief organism_get_fitness
	 * @param p_organism_i
	 * @return
	 */
	real_t organism_get_fitness(uint32_t p_organism_i) const;

	/**
	 * @brief epoch_advance is who make possible the turnover of the population.
	 * In this function every organism die and get replaced with a new one of
	 * new generation that is born with the base genes of the most fittest organisms
	 * of the previous epoch.
	 *
	 * @return true if the advancing was successful
	 */
	bool epoch_advance();

	/**
	 * @brief get_best_personal_fitness returns the best personal fitness ever
	 * used to track the population performances
	 * @return
	 */
	real_t get_best_personal_fitness() const;

	/**
	 * @brief Returns the champion neural network
	 * @param r_brain_area
	 */
	void get_champion_network(brain::SharpBrainArea &r_brain_area) const;

	/**
	 * @brief Returns the statistics of the epoch travelling
	 * @return
	 */
	const NtEpochStatistics &get_epoch_statistics() const;

private:
	/**
	 * @brief speciate splits all organisms in species depending on its
	 * genome compatibility.
	 * The splitting criteria can be controlled by changing the splitting_threshold
	 */
	void speciate();

	/**
	 * @brief kill all species with no organisms
	 */
	void kill_void_species();

	/**
	 * @brief create_species creates a new void species and returns its pointer
	 * @return
	 */
	NtSpecies *create_species();

	/**
	 * @brief destroy_species destroy a species
	 * @param p_species
	 */
	void destroy_species(NtSpecies *p_species);

	/**
	 * @brief destroy_species destroy a species and remove it from the pool
	 * @param p_species
	 * @return the next iterator
	 */
	std::vector<NtSpecies *>::iterator destroy_species(
			std::vector<NtSpecies *>::iterator p_species_iterator);

	/**
	 * @brief destroy_all_species is used to destroy all species
	 */
	void destroy_all_species();

	/**
	 * @brief create a new organism and add it to the pool, return nullptr
	 * if the pool is already full
	 * @return
	 */
	NtOrganism *create_organism();

	/**
	 * @brief destroy the organism and remove it from the organism pool
	 * This version is slower
	 * @param p_organism
	 */
	void destroy_organism(NtOrganism *p_organism);

	/**
	 * @brief destroy the organism and remove it from the organism pool
	 * This version is faster
	 * @param p_organism
	 * @return the next iterator
	 */
	std::vector<NtOrganism *>::iterator destroy_organism(
			std::vector<NtOrganism *>::iterator p_organism_iterator);

	/**
	 * @brief destroy_all_organisms is used to destroy all organisms
	 */
	void destroy_all_organisms();

	/**
	 * @brief This function will kill all organisms marked for death inside the
	 * population organisms
	 */
	void kill_organisms_marked_for_death();

	/**
	 * @brief add_to_species is responsible for the organism addition to the species
	 * @param p_organism
	 * @param p_species
	 */
	void add_organism_to_species(NtOrganism *p_organism, NtSpecies *p_species);

	/**
	 * @brief remove_organism_from_species remove the organism from the assigned specie
	 */
	void remove_organism_from_species(NtOrganism *p_organism);

	/**
	 * @brief get_rand_champion returns the organism champion from a random species
	 * except the one passed through parameter.
	 *
	 * @param p_except_species if null no exception
	 * @return
	 */
	NtOrganism *get_rand_champion(const NtSpecies *p_except_species) const;

private:
	/**
	 * @brief map_rand_gaussian returns the p_x plus a random number
	 * The random number is choosen within a gaussian distribution
	 * @param p_x
	 * @return
	 */
	static real_t rand_gaussian(real_t p_x, void *p_data);

	/**
	 * @brief rand_cold_gaussian does the same thing of the rand_gaussian with
	 * the exception that instead of add this one replace the value
	 * @param p_x
	 * @return
	 */
	static real_t rand_cold_gaussian(real_t p_x, void *p_data);
};

} // namespace brain
