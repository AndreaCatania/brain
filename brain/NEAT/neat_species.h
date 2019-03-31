#pragma once

#include "brain/NEAT/neat_genome.h"
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
	NtPopulation *const owner;

	/**
	 * @brief unique id to identify the specie.
	 * Note: this is different index in the population array.
	 */
	const uint32_t id;

	/**
	 * @brief born_epoch the epoch when this species is born
	 */
	const uint32_t born_epoch;

	/**
	 * @brief The current age of the species
	 */
	int age;

	/**
	 * @brief organisms is the array of organisms of this species, after calling
	 * adjust_fitness this vector get ordered with the most fit as first
	 */
	std::vector<NtOrganism *> organisms;

	/**
	 * @brief champion the champion on this specie
	 * The champion is defined inside the function adjust_fitness
	 */
	NtOrganism *champion;

	/**
	 * @brief average_fitness the average fitness of the species
	 */
	real_t average_fitness;

	/**
	 * @brief higher_fitness_ever store the higher personal fitness ever
	 * registerd for this species.
	 */
	real_t higher_personal_fitness_ever;

	/**
	 * @brief age_of_last_improvement is the age when the higher_fitness_ever is
	 * registered
	 */
	uint32_t age_of_last_improvement;

	/**
	 * @brief stagnant_epochs counts how much epochs are passed from the last
	 * improvement
	 */
	int stagnant_epochs;

	/**
	 * @brief offspring_count is the number of babies of this species
	 */
	int offspring_count;

	/**
	 * @brief champion_offspring_count is used to know how much babies are direct
	 * childs of the champion.
	 *
	 * Be aware of that this value is not added to the offspring_count.
	 */
	int champion_offspring_count;

public:
	/**
	 * @brief NtSpecie constructor
	 * @param p_population
	 * @param p_current_epoch
	 */
	NtSpecies(NtPopulation *p_population, uint32_t p_id, uint32_t p_current_epoch);

	/**
	 * @brief ~NtSpecies is a destructor
	 */
	~NtSpecies();

	/**
	 * @brief returns unique id
	 * @return
	 */
	uint32_t get_id() const;

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
	 * @brief get_born_epoch returns the born epoch
	 * @return
	 */
	uint32_t get_born_epoch() const;

	/**
	 * @brief updates the age
	 * @return
	 */
	void update_age();

	/**
	 * @brief returns the age
	 * @return
	 */
	uint32_t get_age() const;

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
	 * @brief get_stagnant_epochs returns how much stagnant epochs
	 * @return
	 */
	int get_stagnant_epochs() const;

	/**
	 * @brief reset_age_of_last_improvement
	 */
	void reset_age_of_last_improvement();

	/**
	 * @brief set_offspring_count is used to specify how much babies this species
	 * will have
	 * @param p_offspring
	 */
	void set_offspring_count(int p_offspring);

	/**
	 * @brief get_offspring_count returns the expected offspring of this species
	 * @return
	 */
	int get_offspring_count() const;

	/**
	 * @brief set_champion_offspring_count set the amount of babies that born
	 * from the champion.
	 *
	 * It can't be more than the offspring_count, since it tell how much of
	 * offspring is from champion
	 * @param p_offspring
	 */
	void set_champion_offspring_count(int p_offspring);

	/**
	 * @brief get_champion_offspring_count returns the amount of babies born from
	 * the champion
	 * @return
	 */
	int get_champion_offspring_count() const;

	/**
	 * @brief get_champion returns the species champion or null if not yet defined.
	 * The champion is defined inside the function adjust_fitness.
	 * @return
	 */
	NtOrganism *get_champion() const;

	/**
	 * @brief get_average_fitness can be used to get the average fitness of this
	 * species
	 * @return
	 */
	real_t get_average_fitness() const;

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
	 * @param p_survival_ratio
	 */
	void adjust_fitness(
			int p_youngness_age_threshold,
			real_t p_youngness_multiplier,
			int p_stagnant_age_threshold,
			real_t p_stagnant_multiplier,
			real_t p_survival_ratio);

	/**
	 * @brief compute_offspring calculates the offspring of this specie according
	 * to the expected organisms offspring quantity
	 * @param p_remaining is the fractional part that exceded from the previous
	 * species compute_offspring iteration.
	 * @return returns the expected offspring count
	 */
	int compute_offspring(double &r_remaining);

	/**
	 * @brief reproduce will make born new organisms depending on the
	 * expected_offspring, the new organism will be sligtly mutated
	 * compared of the current alive organisms.
	 *
	 * The new organisms will not be assigned to any species because
	 * this task is demanded to the function Population::spetiate()
	 *
	 * This function will mark all its old organisms for death,
	 * that can be deleted using kill_old_organisms.
	 *
	 * @param r_innovations Is the shared list of innovations that happens
	 * in the other species during this epoch transition
	 */
	void reproduce(
			std::vector<NtInnovation> &r_innovations);

	/**
	 * @brief kill all its old organisms
	 */
	void kill_old_organisms();
};

} // namespace brain

/**
 * @brief species_fitness_comparator is used to sort an array of species from
 * the most fittest to last
 * @param
 * @param
 * @return
 */
extern bool species_fitness_comparator(brain::NtSpecies *p_1, brain::NtSpecies *p_2);
