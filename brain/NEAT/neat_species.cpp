#include "neat_species.h"

#include "brain/NEAT/neat_organism.h"
#include "brain/NEAT/neat_population.h"
#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include <algorithm>

brain::NtSpecies::NtSpecies(NtPopulation *p_owner, uint32_t current_epoch) :
		owner(p_owner),
		born_epoch(current_epoch),
		champion(nullptr),
		higher_fitness_ever(0),
		age_of_last_improvement(0),
		offspring_count(0) {
}

brain::NtSpecies::~NtSpecies() {
	ERR_FAIL_COND(organisms.size());
}

void brain::NtSpecies::add_organism(NtOrganism *p_organism) {
	ERR_FAIL_COND(!p_organism);
	ERR_FAIL_COND(p_organism->get_species());
	organisms.push_back(p_organism);
	champion = nullptr;
}

void brain::NtSpecies::remove_organism(const NtOrganism *p_organism) {
	ERR_FAIL_COND(p_organism->get_species() != this);
	auto it = std::find(organisms.begin(), organisms.end(), p_organism);
	if (it != organisms.end())
		organisms.erase(it);
	champion = nullptr;
}

uint32_t brain::NtSpecies::get_born_epoch() const {
	return born_epoch;
}

void brain::NtSpecies::update_age() {
	age = owner->get_epoch() - born_epoch;
}

uint32_t brain::NtSpecies::get_age() const {
	return age;
}

int brain::NtSpecies::size() const {
	return organisms.size();
}

brain::NtOrganism *brain::NtSpecies::get_organism(int p_i) const {
	ERR_FAIL_INDEX_V(p_i, organisms.size(), nullptr);
	return organisms[p_i];
}

int brain::NtSpecies::get_stagnant_epochs() const {
	return stagnant_epochs;
}

void brain::NtSpecies::reset_age_of_last_improvement() {
	age_of_last_improvement = get_age();
	stagnant_epochs = 0;
}

void brain::NtSpecies::set_offspring_count(int p_offspring) {
	offspring_count = p_offspring;
}

int brain::NtSpecies::get_offspring_count() const {
	return offspring_count;
}

void brain::NtSpecies::set_champion_offspring_count(int p_offspring) {
	ERR_FAIL_COND(p_offspring > offspring_count);
	champion_offspring_count = p_offspring;
}

int brain::NtSpecies::get_champion_offspring_count() const {
	return champion_offspring_count;
}

brain::NtOrganism *brain::NtSpecies::get_champion() const {
	return champion;
}

real_t brain::NtSpecies::get_average_fitness() const {
	return average_fitness;
}

void brain::NtSpecies::compute_average_fitness() {
	ERR_FAIL_COND(!organisms.size());
	real_t sum(0);
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		sum += (*it)->get_fitness();
	}
	average_fitness = sum / organisms.size();
}

void brain::NtSpecies::adjust_fitness(
		int p_youngness_age_threshold,
		real_t p_youngness_multiplier,
		int p_stagnant_age_threshold,
		real_t p_stagnant_multiplier,
		real_t p_survival_ratio) {

	update_age();
	stagnant_epochs = age - age_of_last_improvement;

	// Computes forgiving and penalizations
	if (age <= p_youngness_age_threshold) {

		// Still young, give it more chances to do wrong things
		for (auto it = organisms.begin(); it != organisms.end(); ++it) {
			NtOrganism *o = (*it);
			o->set_fitness(o->get_fitness() * p_youngness_multiplier);
		}
	} else if (stagnant_epochs > p_stagnant_age_threshold) {

		// This is not young and also it's stagnant penalize brutally
		for (auto it = organisms.begin(); it != organisms.end(); ++it) {
			NtOrganism *o = (*it);
			o->set_fitness(o->get_fitness() * p_stagnant_multiplier);
		}
	}

	// Shares the fitness
	/// The fitness sharing is a necessary mechanism that allow to
	/// penalize bigger species, by lowering its offspring.
	/// In this way is possible to avoid that a species take over the entire
	/// population and so limit it's diversity and thus its evolution
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		NtOrganism *o = (*it);
		o->set_fitness(o->get_fitness() / organisms.size());
	}

	// Sort organisms, more fit first
	std::sort(organisms.begin(), organisms.end(), organism_fitness_comparator);

	// Check if the species get an improvement
	if (higher_fitness_ever < organisms[0]->get_fitness()) {
		higher_fitness_ever = organisms[0]->get_fitness();
		reset_age_of_last_improvement();
	}

	// Get champion
	champion = organisms[0];

	// Calculates the survival
	uint32_t survival_count = organisms.size() * p_survival_ratio;
	survival_count = MAX(1, survival_count); // 1 should always survive

	ERR_FAIL_COND(survival_count > organisms.size());

	// Mark all others for death
	for (
			auto it = organisms.begin() + survival_count;
			it != organisms.end();
			++it) {

		(*it)->set_mark_for_death(true);
	}
}

int brain::NtSpecies::compute_offspring(double &r_remaining) {
	double expected = r_remaining;

	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		expected += (*it)->get_expected_offspring();
	}

	offspring_count = Math::floor(expected);
	r_remaining = Math::fmod(expected, 1.0);
	return offspring_count;
}

void brain::NtSpecies::reproduce(std::vector<Innovation> &r_innovations) {

	ERR_FAIL_COND(organisms.size() == 0);
	ERR_FAIL_COND(champion_offspring_count > offspring_count);

	// Mark all organisms as dead since they are from the previous generation
	// But don't kill them now since they still need
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		(*it)->set_mark_for_death(true);
	}

	if (!offspring_count)
		return; // Nothing to do

	organisms.reserve(organisms.size() + offspring_count);

	/// Step 1. reproduce the champion offsprings
	/// The last offspring is a perfect clone
	{
		offspring_count -= champion_offspring_count;

		const NtOrganism *champion = *organisms.begin();
		while (champion_offspring_count > 0) {

			NtOrganism *child = owner->create_organism();

			champion->get_genome().duplicate_in(child->get_genome_mutable());

			if (champion_offspring_count > 1) {
				if (Math::randd() < 0.8) {

					// Happens often
					// Mutate link weights
					child->get_genome_mutable().map_link_weights(
							NtPopulation::rand_gaussian,
							owner);
				} else {

					// Happens sometimes
					// Add a link
					const bool add_link_status =
							child->get_genome_mutable().add_random_link(
									owner->settings.genetic_spawn_recurrent_link_threshold,
									r_innovations,
									owner->innovation_number);

					if (!add_link_status) {

						// Almost never happens
						// Was not possible to add a link, so mutates
						// the weights with completelly new weights
						child->get_genome_mutable().map_link_weights(
								NtPopulation::rand_cold_gaussian,
								owner);
					}
				}

			} /*else
				exact copy*/

			--champion_offspring_count;
		}
	}

	/// Step 2. Normal reproduction
	while (offspring_count > 0) {

		// TODO reproduce

		--offspring_count;
	}

	/// Step 3. check phase.
	ERR_FAIL_COND(champion_offspring_count != 0);
	ERR_FAIL_COND(offspring_count != 0);
}

void brain::NtSpecies::kill_old_organisms() {
	auto it = organisms.begin();
	while (it != organisms.end()) {
		if ((*it)->is_marked_for_death()) {
			NtOrganism *o = *it;
			it = organisms.erase(it);
			o->set_species(nullptr);
			delete o;
		} else {
			++it;
		}
	}
}

bool species_fitness_comparator(brain::NtSpecies *p_1, brain::NtSpecies *p_2) {
	return p_1->get_average_fitness() > p_2->get_average_fitness();
}
