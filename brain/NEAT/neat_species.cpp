#include "neat_species.h"

#include "brain/NEAT/neat_organism.h"
#include "brain/NEAT/neat_population.h"
#include "brain/error_macros.h"
#include <algorithm>

brain::NtSpecies::NtSpecies(const NtPopulation *p_owner, uint32_t current_epoch) :
		owner(p_owner),
		born_epoch(current_epoch),
		higher_fitness_ever(0),
		age_of_last_improvement(0) {
}

brain::NtSpecies::~NtSpecies() {
	ERR_FAIL_COND(organisms.size());
}

uint32_t brain::NtSpecies::get_born_epoch() const {
	return born_epoch;
}

uint32_t brain::NtSpecies::get_age() const {
	return owner->get_epoch() - born_epoch;
}

void brain::NtSpecies::add_organism(NtOrganism *p_organism) {
	ERR_FAIL_COND(!p_organism);
	ERR_FAIL_COND(p_organism->get_species());
	organisms.push_back(p_organism);
}

void brain::NtSpecies::remove_organism(const NtOrganism *p_organism) {
	ERR_FAIL_COND(p_organism->get_species() != this);
	auto it = std::find(organisms.begin(), organisms.end(), p_organism);
	if (it != organisms.end())
		organisms.erase(it);
}

int brain::NtSpecies::size() const {
	return organisms.size();
}

brain::NtOrganism *brain::NtSpecies::get_organism(int p_i) const {
	ERR_FAIL_INDEX_V(p_i, organisms.size(), nullptr);
	return organisms[p_i];
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
		real_t p_stagnant_multiplier) {

	const int age = get_age();
	const int stagnant_time = age - age_of_last_improvement;

	// Computes forgiving and penalizations
	if (age <= p_youngness_age_threshold) {

		// Still young, give it more chances to do wrong things
		for (auto it = organisms.begin(); it != organisms.end(); ++it) {
			NtOrganism *o = (*it);
			o->set_fitness(o->get_fitness() * p_youngness_multiplier);
		}
	} else if (stagnant_time > p_stagnant_age_threshold) {

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
		age_of_last_improvement = age;
	}

	// TODO genetics.cpp:2724
	int a = 0;
}
