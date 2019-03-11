#pragma once

#include "brain/math/math_defs.h"
#include "brain/typedefs.h"
#include <vector>

namespace brain {

class NtPopulation;
class NtOrganism;

/**
 * @brief The NtSpecie class represent a group of organisms that have the genomes
 * similar, this mean that they share the same traits.
 *
 * The organisms are grouped in species in order to make they compete within the
 * specie organisms. By doing so it's possible to choose the best organism in a
 * poll of organisms with same traits.
 *
 * If instead we would allow an organism to compete directly with the entire
 * population we could lose some organisms, with particular trait not yet fully
 * formed, prematurely.
 */
class NtSpecies {

	/**
	 * @brief owner of this specie
	 */
	const NtPopulation *owner;

	/**
	 * @brief born_epoch the epoch when this species is born
	 */
	const uint32_t born_epoch;

	/**
	 * @brief organisms is the array of organisms of this specie
	 */
	std::vector<NtOrganism *> organisms;

	/**
	 * @brief average_fitness the average fitness of the species
	 */
	real_t average_fitness;

	/**
	 * @brief higher_fitness_ever store the higher fitness ever registerd for
	 * this species
	 */
	real_t higher_fitness_ever;

	/**
	 * @brief age_of_last_improvement is the age when the higher_fitness_ever is
	 * registered
	 */
	uint32_t age_of_last_improvement;

public:
	/**
	 * @brief NtSpecie constructor
	 * @param p_population
	 * @param p_current_epoch
	 */
	NtSpecies(const NtPopulation *p_population, uint32_t p_current_epoch);

	/**
	 * @brief ~NtSpecies is a destructor
	 */
	~NtSpecies();

	/**
	 * @brief get_born_epoch returns the born epoch
	 * @return
	 */
	uint32_t get_born_epoch() const;

	/**
	 * @brief get_age compute the age and returns it
	 * @return
	 */
	uint32_t get_age() const;

	/**
	 * @brief add_organism add new organism to this specie
	 * @param p_organism
	 */
	void add_organism(NtOrganism *p_organism);

	/**
	 * @brief remove_organism from this specie
	 * @param p_organism
	 */
	void remove_organism(const NtOrganism *p_organism);

	/**
	 * @brief size returns the organisms count
	 * @return
	 */
	int size() const;

	/**
	 * @brief get_organism get organism in the i position
	 * @param p_i
	 * @return
	 */
	NtOrganism *get_organism(int p_i) const;

	/**
	 * @brief compute_average_fitness compute the average fitness
	 */
	void compute_average_fitness();

	/**
	 * @brief adjust_fitness will change the fitness of its organisms
	 * depending os these criterias:
	 *
	 * If the species is young the fitness is increased to protect its organisms
	 * from a premature death, on the other hand it will lower drastically the
	 * fitness if this species doesn't improved from a certain period.
	 *
	 * @param p_youngness_age_threshold
	 * @param p_youngness_multiplier
	 * @param p_stagnant_age_threshold
	 * @param p_stagnant_multiplier
	 */
	void adjust_fitness(
			int p_youngness_age_threshold,
			real_t p_youngness_multiplier,
			int p_stagnant_age_threshold,
			real_t p_stagnant_multiplier);
};

} // namespace brain
