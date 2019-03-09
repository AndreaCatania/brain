#pragma once

#include "brain/NEAT/neat_genome.h"
#include "brain/brain_areas/sharp_brain_area.h"

namespace brain {

class NtPopulation;

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
	 * @brief get_genome_mutable give the possibility to mutate the
	 * genome from outside
	 * @return
	 */
	NtGenome &get_genome_mutable();
};

} // namespace brain
