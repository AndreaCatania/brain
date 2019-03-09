#pragma once

#include "brain/NEAT/neat_genome.h"
#include "brain/NEAT/neat_organism.h"
#include <random>

namespace brain {

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
	 * @brief splitting_threshold is used to know if two organism are of the
	 * same specie.
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
	real_t splitting_threshold;

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
	 * @brief organisms are the list of all organism of this population
	 */
	std::vector<NtOrganism> organisms;

public:
	/**
	 * @brief NtPopulation construct the population by spawning each member
	 * that shares the traits with the ancestor_genome.
	 *
	 * This mean that the population will have the same structure of the ancestor
	 * with link weight mutated
	 *
	 * @param p_ancestor_genome
	 * @param p_population_size
	 * @param p_splitting_threshold
	 * @param p_seed this is used to change the behaviour of the population
	 * @param p_learning_deviation is a parameter used to control the learning
	 * delta during the weight changing. Check the Gaussian random to understand
	 * better its use.
	 */
	NtPopulation(
			const NtGenome &p_ancestor_genome,
			int p_population_size,
			real_t p_splitting_threshold = 0.3f,
			uint64_t p_seed = 1,
			real_t p_learning_deviation = 1.f);

private:
	/**
	 * @brief speciate splits all organisms in species depending on its
	 * genome compatibility.
	 * The splitting criteria can be controlled by changing the splitting_threshold
	 */
	void speciate();

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
