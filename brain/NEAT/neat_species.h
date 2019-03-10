#pragma once

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
	 * @brief organisms is the array of organisms of this specie
	 */
	std::vector<NtOrganism *> organisms;

public:
	/**
	 * @brief NtSpecie constructor
	 * @param p_population
	 */
	NtSpecies(const NtPopulation *p_population);

	/**
	 * @brief ~NtSpecies is a destructor
	 */
	~NtSpecies();

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
};

} // namespace brain
