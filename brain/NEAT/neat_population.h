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
	real_t learning_deviation = 1.f;

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
	real_t genetic_compatibility_threshold = 0.3f;

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
};

/**
 * @brief The NtPopulation class is responsible for the born, grow, death of
 * each member.
 *
 * This class is the API of the NEAT.
 */
class NtPopulation {

	/**
	 * @brief innovation_number is an incremental number that is used to
	 * mark and so track all changes to the organism genome.
	 */
	uint32_t innovation_number;

	/**
	 * @brief genetic_compatibility_threshold is used to determine the species
	 * organisms, during genetic comparison.
	 */
	real_t genetic_compatibility_threshold;

	/**
	 * @brief genetic_disjoints_significance is used during the genetic comparison
	 * between two genomes.
	 */
	real_t genetic_disjoints_significance;

	/**
	 * @brief genetic_excesses_significance is used during the genetic comparison
	 * between two genomes.
	 */
	real_t genetic_excesses_significance;

	/**
	 * @brief genetic_weights_significance is used during the genetic comparison
	 * between two genomes.
	 */
	real_t genetic_weights_significance;

	/**
	 * @brief fitness_exponent is used to scale the fitness exponentially and thus
	 * differentiate more the organisms when the fitness increase.
	 *
	 * This is useful because in some situations when the network is already optimized
	 * the fitness gain is low even if there are some benefits that is worth to notice.
	 */
	real_t fitness_exponent;

	/**
	 * @brief species_youngness_age_threshold the ages within a species is considered
	 * young and so is protected
	 */
	int species_youngness_age_threshold;

	/**
	 * @brief species_youngness_multiplier the fitness multiplier applied to the
	 * young species
	 */
	real_t species_youngness_multiplier;

	/**
	 * @brief species_stagnant_age_threshold the ages without improvements
	 * to consider a species stagnant
	 */
	int species_stagnant_age_threshold;

	/**
	 * @brief species_stagnant_multiplier the penalty multiplier applied to the
	 * stagnant species
	 */
	real_t species_stagnant_multiplier;

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
	 * @brief get_organism_count returns the count of population organisms
	 * @return
	 */
	uint32_t get_organism_count() const;

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
	 * @brief epoch_advance is the function that depending on the fitness
	 * decide to replace the less performant organisms with the offspring of
	 * most performant organisms called champions.
	 */
	void epoch_advance();

private:
	/**
	 * @brief speciate splits all organisms in species depending on its
	 * genome compatibility.
	 * The splitting criteria can be controlled by changing the splitting_threshold
	 */
	void speciate();

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
	 * @brief destroy_all_species is used to destroy all species
	 */
	void destroy_all_species();

	/**
	 * @brief destroy_all_organism is used to destroy all organism
	 */
	void destroy_all_organism();

	/**
	 * @brief add_to_specie is responsible for the organism addition to the species
	 * @param p_organism
	 * @param p_species
	 */
	void add_organism_to_specie(NtOrganism *p_organism, NtSpecies *p_species);

	/**
	 * @brief remove_specie remove the organism from the assigned specie
	 */
	void remove_organism_from_specie(NtOrganism *);

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
