#pragma once

#include "brain/NEAT/neat_genome.h"
#include <random>

namespace brain {

class NtSpecies;
class NtOrganism;

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
	 * @brief learning_deviation is a parameter used to control the learning
	 * delta during the weight changing. Check the Gaussian random to understand
	 * better its use.
	 */
	real_t learning_deviation = 0.75f;

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
	real_t genetic_compatibility_threshold = 0.5f;

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
	real_t genetic_spawn_recurrent_link_threshold = 0.35;

	/**
	 * @brief genetic_mate_prob is used to define the probability for
	 * a genome to mate with another one, instead to mutate.
	 */
	real_t genetic_mate_prob = 0.5;

	/**
	 * @brief genetic_mate_inside_species_threshold is used to controll where the
	 * other parent (dad) should be taken within the same species of the mom or
	 * outside.
	 */
	real_t genetic_mate_inside_species_threshold = 0.5;

	/**
	 * @brief genetic_mating_* are used to decide what type of crossover operation
	 * should occur more often.
	 *
	 * These parameters will be normalized (If all are the same they will have
	 * the same probability)
	 */
	real_t genetic_mating_multipoint_threshold = 0.4;
	real_t genetic_mating_multipoint_avg_threshold = 0.4;
	real_t genetic_mating_singlepoint_threshold = 0.2;

	/**
	 * @brief genetic_mutate_* are used to decide what type of mutation should
	 * occur more often.
	 *
	 * These parameters will be normalized (If all are the same they will have
	 * the same probability)
	 */
	real_t genetic_mutate_add_link_porb = 0.3;
	real_t genetic_mutate_add_node_prob = 0.15;
	real_t genetic_mutate_link_weight_prob = 0.5;
	real_t genetic_mutate_toggle_link_enable_prob = 0.05;

	/**
	 * @brief fitness_exponent is used to scale the fitness exponentially and thus
	 * differentiate more the organisms when the fitness increase.
	 *
	 * This is useful because in some situations when the network is already optimized
	 * the fitness gain is low even if there are some benefits that is worth to notice.
	 */
	real_t fitness_exponent = 2.f;

	/**
	 * @brief species_youngness_age_threshold the ages within a species is considered
	 * young and so is protected
	 */
	int species_youngness_age_threshold = 10;

	/**
	 * @brief species_youngness_multiplier the fitness multiplier applied to the
	 * young species
	 */
	real_t species_youngness_multiplier = 3.0f;

	/**
	 * @brief species_stagnant_age_threshold the ages without improvements
	 * to consider a species stagnant
	 */
	int species_stagnant_age_threshold = 10;

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
	 * @brief species_stealing_protection_age_threshold is used to protect the
	 * species from stealing even if they perform low until they pass this threshold
	 */
	int species_stealing_protection_age_threshold = 5;

	/**
	 * @brief species_stealing_limit is used to stop the stealing when they get
	 * below this value.
	 * In thi way is possible to protect the specie from the death due to the
	 * stealing.
	 */
	int species_stealing_limit = 2;

	/**
	 * @brief population_stagnant_age_thresold is used to detect if the population
	 * become stagnant and is not able to improve more.
	 */
	int population_stagnant_age_thresold = 15;

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
	int population_cribs_stealing = 10;
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
	 * @brief organism_add_fitness is used to tell how this organism is doing well
	 * @param p_organism_i
	 * @param p_fitness
	 */
	void organism_add_fitness(uint32_t p_organism_i, real_t p_fitness) const;

	/**
	 * @brief epoch_advance is who make possible the turnover of the population.
	 * In this function every organism die and get replaced with a new one of
	 * new generation that is born with the base genes of the most fittest organisms
	 * of the previous epoch.
	 * @return true if the advancing was successful
	 */
	bool epoch_advance();

	/**
	 * @brief get_best_personal_fitness returns the best personal fitness ever
	 * used to track the population performances
	 * @return
	 */
	real_t get_best_personal_fitness() const;

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
