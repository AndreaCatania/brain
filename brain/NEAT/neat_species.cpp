#include "neat_species.h"

#include "brain/NEAT/neat_organism.h"
#include "brain/NEAT/neat_population.h"
#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include <algorithm>

brain::NtSpecies::NtSpecies(NtPopulation *p_owner, uint32_t current_epoch) :
		owner(p_owner),
		born_epoch(current_epoch),
		age(0),
		champion(nullptr),
		average_fitness(0),
		higher_personal_fitness_ever(0),
		age_of_last_improvement(0),
		stagnant_epochs(0),
		offspring_count(0),
		champion_offspring_count(0) {
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
			o->set_fitness(o->get_personal_fitness() * p_youngness_multiplier);
		}
	} else if (stagnant_epochs > p_stagnant_age_threshold) {

		// This is not young and also it's stagnant penalize brutally
		for (auto it = organisms.begin(); it != organisms.end(); ++it) {
			NtOrganism *o = (*it);
			o->set_fitness(o->get_personal_fitness() * p_stagnant_multiplier);
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

	// Get champion
	champion = organisms[0];

	// Check if the species get an improvement
	if (higher_personal_fitness_ever < champion->get_personal_fitness()) {
		higher_personal_fitness_ever = champion->get_personal_fitness();
		reset_age_of_last_improvement();
	}

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

void brain::NtSpecies::reproduce(
		std::vector<NtInnovation> &r_innovations) {

	ERR_FAIL_COND(organisms.size() == 0);
	ERR_FAIL_COND(champion_offspring_count > offspring_count);

	// Mark all organisms as dead since they are from the previous generation
	// But don't kill them now since they still need
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		(*it)->set_mark_for_death(true);
	}

	if (!offspring_count)
		return; // Nothing to do

	bool is_champion_cloned = false;
	const NtOrganism *champion = *organisms.begin();

	/// Step 1. reproduce the champion offsprings
	/// The last offspring is a perfect clone
	{
		offspring_count -= champion_offspring_count;

		while (champion_offspring_count > 0) {

			NtOrganism *child = owner->create_organism();

			champion->get_genome().duplicate_in(child->get_genome_mutable());

			if (champion_offspring_count > 1) {
				if (Math::randd() < 0.8) {

					child->log += "\nCHAMPION MUTATE Add Weight";
					// Happens more often
					// Mutate link weights
					child->get_genome_mutable().mutate_random_link_weight(
							NtPopulation::rand_gaussian,
							owner);
				} else {

					child->log += "\nCHAMPION MUTATE Add random link";
					// Happens sometimes
					// Add a link
					const bool add_link_status =
							child->get_genome_mutable().mutate_add_random_link(
									owner->settings.genetic_mutate_add_link_recurrent_prob,
									r_innovations,
									owner->innovation_number);

					if (!add_link_status) {

						// Almost never happens
						// Was not possible to add a link, so mutates
						// the weights with completelly new weights
						child->get_genome_mutable().mutate_random_link_weight(
								NtPopulation::rand_cold_gaussian,
								owner);
					}
				}

			} else {
				// Just an exact copy of the champion
				is_champion_cloned = true;
			}

			--champion_offspring_count;
		}
	}

	/// Step 2. Check if the champion deserve a clone
	if (!is_champion_cloned && 4 < offspring_count) {
		is_champion_cloned = true;

		NtOrganism *child = owner->create_organism();
		champion->get_genome().duplicate_in(child->get_genome_mutable());

		--offspring_count;
	}

	/// Step 3. Normal reproduction

	const int organisms_last_index(organisms.size() - 1);

	const real_t mating_prob = owner->settings.genetic_mate_prob;
	const real_t mating_multipoint_prob = owner->settings.genetic_mate_multipoint_threshold;
	const real_t mating_multipoint_avg_prob = owner->settings.genetic_mate_multipoint_avg_threshold;
	const real_t mating_singlepoint_prob = owner->settings.genetic_mate_singlepoint_threshold;
	const real_t mutate_add_link_prob = owner->settings.genetic_mutate_add_link_porb;
	const real_t mutate_add_node_prob = owner->settings.genetic_mutate_add_node_prob;
	const real_t mutate_link_weight_prob = owner->settings.genetic_mutate_link_weight_prob;
	const real_t mutate_toggle_link_enable_prob = owner->settings.genetic_mutate_toggle_link_enable_prob;

	// Create ranges
	const real_t mating_tot =
			mating_multipoint_prob +
			mating_multipoint_avg_prob +
			mating_singlepoint_prob;

	const real_t m_m_range = mating_multipoint_prob / mating_tot;
	const real_t m_m_a_range = (mating_multipoint_avg_prob / mating_tot) + m_m_range;

	const real_t mutate_tot =
			mutate_add_link_prob +
			mutate_add_node_prob +
			mutate_link_weight_prob +
			mutate_toggle_link_enable_prob;

	const real_t m_a_l_range = mutate_add_link_prob / mutate_tot;
	const real_t m_a_n_range = (mutate_add_node_prob / mutate_tot) + m_a_l_range;
	const real_t m_l_w_range = (mutate_link_weight_prob / mutate_tot) + m_a_n_range;

	while (offspring_count > 0) {

		NtOrganism *child = owner->create_organism();
		ERR_FAIL_COND(!child);

		const int mom_index = static_cast<int>(Math::random(0, organisms_last_index) + 0.5);
		NtOrganism *mom = organisms[mom_index];

		bool state = false;

		if (Math::randd() < mating_prob && organisms_last_index > 0) {

			// Mate

			NtOrganism *dad = nullptr;

			if (Math::randd() >= owner->settings.genetic_mate_inside_species_threshold) {
				// Select the champion of a random species to be the dad
				dad = owner->get_rand_champion(this);
			}

			if (!dad) {
				/// This can occurs even if the champion is taken randomly from
				/// outside the species, with this I'm sure that the dad is nevel
				/// null

				// Select the dad from the same species
				const int dad_index = static_cast<int>(Math::random(0, organisms_last_index) + 0.5);
				dad = organisms[dad_index];
			}

			const real_t r(Math::randd());
			if (r < m_m_range) {

				child->log += "\nMATE multipoint";

				// Multipoint mating
				state = child->get_genome_mutable().mate_multipoint(
						mom->get_genome(),
						mom->get_personal_fitness(),
						dad->get_genome(),
						dad->get_personal_fitness(),
						false);

			} else if (r < m_m_a_range) {

				child->log += "\nMATE multipoint avg";
				// Multipoint Average mating
				state = child->get_genome_mutable().mate_multipoint(
						mom->get_genome(),
						mom->get_personal_fitness(),
						dad->get_genome(),
						dad->get_personal_fitness(),
						true);

			} else {

				child->log += "\nMATE singlepoint";
				// Singlepoint mating
				state = child->get_genome_mutable().mate_singlepoint(
						mom->get_genome(),
						dad->get_genome());
			}

		} else {

			// Mutate

			mom->get_genome().duplicate_in(child->get_genome_mutable());

			const real_t r(Math::randd());
			if (r < m_a_l_range) {

				child->log += "\nMUTATE add link";
				// Mutate add link
				state = child->get_genome_mutable().mutate_add_random_link(
						owner->settings.genetic_mutate_add_link_recurrent_prob,
						r_innovations,
						owner->innovation_number);

			} else if (r < m_a_n_range) {

				child->log += "\nMUTATE add neuron";
				// Mutate add neuron
				state = child->get_genome_mutable().mutate_add_random_neuron(
						r_innovations,
						owner->innovation_number);

			} else if (r < m_l_w_range) {

				child->log += "\nMUTATE mutate weight";
				// Mutate link weight
				if (Math::randd() < owner->settings.genetic_mutate_link_weight_uniform_prob) {

					// TODO use all or random???
					child->get_genome_mutable().mutate_all_link_weights(
							NtPopulation::rand_gaussian,
							owner);
				} else {

					// TODO use all or random???
					child->get_genome_mutable().mutate_all_link_weights(
							NtPopulation::rand_cold_gaussian,
							owner);
				}
				state = true;
			} else {

				child->log += "\nMUTATE toggle link activation";
				// Mutate toggle link enabled
				child->get_genome_mutable().mutate_random_link_toggle_activation();
				state = true;
			}
		}

		if (!state) {
			// This could happen since Sometimes is not possible to mutate the
			// organism
			// TODO just put something there to track this
			// WARN_PRINTS("Somthing went wrong during the organism reproduction.");
		}

		if (!child->get_genome().check_innovation_numbers())
			DEBUG_ONLY(ERR_FAIL_COND(!child->get_genome().check_innovation_numbers()));

		--offspring_count;
	}

	/// Step 4. check phase.
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
