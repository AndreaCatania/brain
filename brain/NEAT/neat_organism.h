#pragma once

#include "brain/NEAT/neat_genome.h"
#include "brain/brain_areas/sharp_brain_area.h"

namespace brain {

class NtPopulation;
class NtSpecies;

/**
 * @brief The NtOrganism class represent the actual organism with all info
 * like the phenotype and the genome
 */
class NtOrganism {

	/**
	 * @brief owner is the population where this organism belong
	 */
	const NtPopulation *owner;

	/**
	 * @brief specie where this organism belong. If null mean that it is not part
	 * of any specie yet
	 */
	NtSpecies *species;

	/**
	 * @brief The genome holds all information to create the brain area
	 */
	NtGenome genome;

	/**
	 * @brief brain_area is the phenotype and is created following the genome
	 * instructions.
	 */
	SharpBrainArea brain_area;

	/**
	 * @brief dead is used to know if the current organism is ready to be used
	 */
	bool dead;

	/**
	 * @brief is_dirty_brain_area is used to know when the brain_area doesn't
	 * match anymore the genome and it must be recreated.
	 */
	bool is_dirty_brain_area;

public:
	/**
	 * @brief NtOrganism constructor
	 */
	NtOrganism(const NtPopulation *p_owner);

	/**
	 * @brief NtOrganism Destructor
	 */
	~NtOrganism();

	/**
	 * @brief get_genome_mutable give the possibility to mutate the
	 * genome from outside
	 * @return
	 */
	NtGenome &get_genome_mutable();

	/**
	 * @brief get_genome get the genome
	 * @return
	 */
	const NtGenome &get_genome() const;

	/**
	 * @brief set_species set the species where this organism belongs
	 * @param p_specie
	 */
	void set_species(NtSpecies *p_species);

	/**
	 * @brief get_species returns the speciews where this organism belongs
	 * @return
	 */
	NtSpecies *get_species() const;
};

} // namespace brain
