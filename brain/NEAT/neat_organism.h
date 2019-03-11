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
	mutable SharpBrainArea brain_area;

	/**
	 * @brief dead is used to know if the current organism is ready to be used
	 */
	bool dead;

	/**
	 * @brief is_dirty_brain_area is used to know when the brain_area doesn't
	 * match anymore the genome and it must be recreated.
	 */
	mutable bool is_dirty_brain_area;

	/**
	 * @brief middle_fitness_sum is used to store all fitness durin this epoch.
	 * It will be used to take the fitness average
	 */
	real_t middle_fitness_sum;

	/**
	 * @brief middle_fitness_count is counting how much fitness are set.
	 * It will be used to take the fitness average
	 */
	uint32_t middle_fitness_count;

	/**
	 * @brief fitness the final fitness of the organism that is calculated during the
	 * epoch advancing
	 */
	real_t fitness;

	/**
	 * @brief personal_fitness is the original fitness of the organism and can't
	 * be influenced by the species where it belong
	 */
	real_t personal_fitness;

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
	 * @brief get_brain_area get neural network
	 * @return
	 */
	const SharpBrainArea &get_brain_area() const;

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

	/**
	 * @brief add_middle_fitness add middle fitness.
	 * This function allow to submit the fitness per guess, in this way
	 * By storying the performances history is possible to evaluete it better
	 * during the epoch advancing
	 * @param p_fitness
	 */
	void add_middle_fitness(real_t p_fitness);

	/**
	 * @brief clear_middle_fitness reset the fitness_sum and fitness_count
	 */
	void clear_middle_fitness();

	/**
	 * @brief compute_final_fitness is where the fitness is finally computed
	 * The exponent parameter is used to increase the fitness exponentially and
	 * so differentiate more the organisms performances.
	 * Just pass 1 if you don't want to increase it exponentially
	 * @param p_exponent
	 */
	void compute_final_fitness(real_t p_exponent);

	/**
	 * @brief set_fitness set the fitness
	 * @param p_fitness
	 */
	void set_fitness(real_t p_fitness);

	/**
	 * @brief get_fitness returns the fitness.
	 * @return
	 */
	real_t get_fitness() const;

	/**
	 * @brief get_personal_fitness returns the personal and untouched fitness.
	 * @return
	 */
	real_t get_personal_fitness() const;
};

} // namespace brain

/**
 * @brief organism_fitness_comparator used to sort the organisms from most fit to less
 * @param p_1
 * @param p_2
 * @return
 */
extern bool organism_fitness_comparator(brain::NtOrganism *p_1, brain::NtOrganism *p_2);
