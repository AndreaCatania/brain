#include "neat_population.h"

#include "brain/NEAT/neat_genetic.h"
#include "brain/NEAT/neat_organism.h"
#include "brain/NEAT/neat_species.h"
#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include <algorithm>

brain::NtPopulation::NtPopulation(
		const NtGenome &p_ancestor_genome,
		int p_population_size,
		NtPopulationSettings &p_settings) :
		population_size(p_population_size),
		settings(p_settings),
		species_last_index(0),
		rand_generator(p_settings.seed),
		gaussian_distribution(0, p_settings.learning_deviation),
		epoch(1),
		best_personal_fitness(0.f),
		epoch_last_improvement(epoch) {

	organisms.reserve(p_population_size);

	for (int i = 0; i < population_size; ++i) {

		NtOrganism *o = create_organism();

		NtGenome &new_organism_genome = o->get_genome_mutable();
		p_ancestor_genome.duplicate_in(new_organism_genome);

		new_organism_genome.mutate_all_link_weights(
				rand_gaussian,
				static_cast<void *>(this));
	}

	innovation_number = p_ancestor_genome.get_innovation_number();

	speciate();
}

brain::NtPopulation::~NtPopulation() {
	destroy_all_organisms();
	destroy_all_species();
}

uint32_t brain::NtPopulation::get_epoch() const {
	return epoch;
}

uint32_t brain::NtPopulation::get_population_size() const {
	return population_size;
}

const brain::SharpBrainArea *brain::NtPopulation::organism_get_network(uint32_t p_organism_i) const {
	ERR_FAIL_INDEX_V(p_organism_i, population_size, nullptr);
	return &organisms[p_organism_i]->get_brain_area();
}

void brain::NtPopulation::organism_set_fitness(uint32_t p_organism_i, real_t p_fitness) const {
	ERR_FAIL_INDEX(p_organism_i, population_size);
	organisms[p_organism_i]->set_evaluation(p_fitness);
}

bool brain::NtPopulation::epoch_advance() {

	statistics.clear();
	statistics.epoch = epoch;

	++epoch;

	/// Step 1. Take the best organism and make a copy of its genome
	NtOrganism *population_champion = *organisms.begin();
	{
		// This algorithm prefer the oldest genome when the fitness is the same
		for (auto it_o = organisms.begin(); it_o != organisms.end(); ++it_o) {
			bool is_better = false;
			if (ABS(population_champion->get_personal_fitness() - (*it_o)->get_personal_fitness()) <= CMP_EPSILON) {

				// They have the same personal fitness

				if (population_champion->get_species()->get_id() == (*it_o)->get_species()->get_id()) {
					// They are the same species

					if ((*it_o)->is_champion_clone()) {
						is_better = true;
					}
				} else {
					// They are different species so take the oldest
					is_better = population_champion->get_species()->get_born_epoch() > (*it_o)->get_species()->get_born_epoch();
				}
			} else {
				is_better = population_champion->get_personal_fitness() < (*it_o)->get_personal_fitness();
			}

			if (is_better) {
				population_champion = *it_o;
			}
		}
	}
	ERR_FAIL_COND_V(!population_champion, false);
	population_champion->get_genome().duplicate_in(champion_genome);

	population_champion->set_the_best(true);

	statistics.pop_champion_fitness = population_champion->get_personal_fitness();
	statistics.pop_champion_species_id = population_champion->get_species()->get_id();

	/// Step 2. Compute species average fitness, then adjust it
	int ages_sum(0);
	for (auto it_s = species.begin(); it_s != species.end(); ++it_s) {
		(*it_s)->compute_average_fitness();
		(*it_s)->adjust_fitness(
				settings.species_youngness_age_threshold,
				settings.species_youngness_multiplier,
				settings.species_stagnant_age_threshold,
				settings.species_stagnant_multiplier,
				settings.species_survival_ratio);

		ages_sum += (*it_s)->get_age();
	}

	statistics.species_count = species.size();
	statistics.species_avg_ages = ages_sum / species.size();

	std::vector<NtSpecies *> ordered_species(species.size());
	std::copy(species.begin(), species.end(), ordered_species.begin());

	std::sort(
			ordered_species.begin(),
			ordered_species.end(),
			species_comparator);

	NtSpecies *best_species = *ordered_species.begin();

	statistics.species_best_id = best_species->get_id();
	statistics.species_best_age = best_species->get_age();

	/// Stage 3. calculates average population fitness
	real_t total_fitness(0);
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		total_fitness += (*it)->get_fitness();
	}
	real_t population_average_fitness = total_fitness / population_size;

	statistics.pop_avg_fitness = population_average_fitness;

	/// Step 4. Calculates the number of offspring for each organism
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		(*it)->set_expected_offspring((*it)->get_fitness() / population_average_fitness);
	}

	/// Step 5. Calculates the offspring per species
	{
		double remaining(0);
		int total_expected_offsprings(0);
		for (auto it = species.begin(); it != species.end(); ++it) {
			total_expected_offsprings += (*it)->compute_offspring(remaining);
		}

		// Check if the expected offsprings are less than expected due to
		// the precision loss
		if (total_expected_offsprings < population_size) {
			best_species->set_offspring_count(
					best_species->get_offspring_count() + 1);
			++total_expected_offsprings;

			/// When the total_expected still below
			/// could mean that there is a bug somewhere
			/// or the best species is a stagnant species and it get killed by its age
			/// What happens is that the population rebord from the best specie
			if (total_expected_offsprings < population_size) {
				WARN_PRINTS("The expected offsprings are below normality, The population is going to rebord from the best population.");
				for (auto it = species.begin(); it != species.end(); ++it) {
					(*it)->set_offspring_count(0);
				}
				best_species->set_offspring_count(population_size);
				total_expected_offsprings = population_size;
			}
		}

		ERR_FAIL_COND_V(total_expected_offsprings != population_size, false);
	}

	statistics.species_best_offspring_pre_steal = best_species->get_offspring_count();

	/// Step 6. Perform offspring re-assignment
	/// This phase changes depending if the population is stagnant or not
	/// When the pop is not stagnant and is allowed to stole cribs from
	/// the lowest species the stoling process is performed

	if (best_personal_fitness < population_champion->get_personal_fitness()) {
		best_personal_fitness = population_champion->get_personal_fitness();
		epoch_last_improvement = epoch;
	}

	statistics.pop_epoch_last_improvement = epoch_last_improvement;

	if (epoch - epoch_last_improvement > settings.population_stagnant_age_thresold) {

		statistics.pop_is_stagnant = true;

		/// The population is stagnant
		/// So try to restart it from the best species

		if (ordered_species.size() < 2) {

			WARN_PRINTS("The population is stagnant, and there is only one species. Trying to repopulate, but the expectation is low.");

			best_species->set_offspring_count(population_size);
			best_species->set_champion_offspring_count(population_size);
			best_species->reset_age_of_last_improvement();

		} else {

			WARN_PRINTS("The population is stagnant, Repopulates from the champions of the best two species.");

			NtSpecies *first_species = *(ordered_species.begin() + 0);
			NtSpecies *second_species = *(ordered_species.begin() + 1);

			const int population_half_size = population_size * 0.5;
			// Avoid precision loss during above division operation
			first_species->set_offspring_count(population_size - population_half_size);
			second_species->set_offspring_count(population_half_size);

			first_species->set_champion_offspring_count(population_size - population_half_size);
			second_species->set_champion_offspring_count(population_half_size);

			// Avoid to become stagnant
			first_species->reset_age_of_last_improvement();
			second_species->reset_age_of_last_improvement();

			// Make sure that other species doen't reproduce
			if (ordered_species.size() > 2) {
				for (auto it = ordered_species.begin() + 2;
						it != ordered_species.end();
						++it) {
					(*it)->set_offspring_count(0);
				}
			}
		}

		epoch_last_improvement = epoch;

	} else if (settings.cribs_stealing && ordered_species.size() > 2) {

		/// Stole the cribs from the worst species
		/// Then reassign it following this criterias:
		///  1/5 are given to the two not dying best species
		///  1/10 are give to the third not dying best species
		///  The rest is assigned to all other not dying species randomly
		///  If remains yet cribs to allocate, will be given them all to the best species

		int stolen_cribs(0);

		// Iterates from the botton and skip the champion species
		for (auto it = ordered_species.rbegin(); it != ordered_species.rend(); ++it) {
			NtSpecies *s = *it;

			if (ordered_species[1] == s || ordered_species[0] == s) {
				// Avoid to steal from the two best species
				break;
			}

			if (s->get_age() > settings.cribs_stealing_protection_age_threshold &&
					s->get_offspring_count() > settings.cribs_stealing_limit) {

				// Steal from this species most possible
				const int possible_booty = s->get_offspring_count() -
										   settings.cribs_stealing_limit;

				const int wanted_booty = settings.cribs_stealing -
										 stolen_cribs;

				const int booty = MIN(possible_booty, wanted_booty);

				s->set_offspring_count(s->get_offspring_count() - booty);
				stolen_cribs += booty;

				ERR_FAIL_COND_V(stolen_cribs > settings.cribs_stealing, false);
				if (stolen_cribs == settings.cribs_stealing) {
					break;
				}
			}
		}

		statistics.pop_stolen_cribs = stolen_cribs;

		// Assign stolen cribs to the first second and third species
		const int stolen_one_fifth = stolen_cribs / 5;
		const int stolen_one_tenth = stolen_cribs / 10;

		{
			int i(0);
			for (auto it = ordered_species.begin();
					it != ordered_species.end() && i < 3;
					++it) {

				// Avoid to add it to dying species
				if ((*it)->get_stagnant_epochs() > settings.species_stagnant_age_threshold)
					continue;

				const int assignment = i == 0 ? stolen_one_fifth : stolen_one_tenth;

				(*it)->set_offspring_count(
						(*it)->get_offspring_count() + assignment);

				(*it)->set_champion_offspring_count(
						(*it)->get_champion_offspring_count() + assignment);

				stolen_cribs -= assignment;
				++i;
			}
		}

		// Assign the rest using a roulet to all other non dying species
		{

			const int roulet_spots = ordered_species.end() - ordered_species.begin();

			if (roulet_spots) {

				// Thanks to this is possible to give always the same percentage
				// of possibility to the all species no matter the quantity
				const double roulet_threshold = 1. / roulet_spots;

				// This is used as fallback to stop the roulet
				int spin_left = stolen_cribs * 4;

				real_t roulet_spot_right(0);

				while (stolen_cribs > 0 && spin_left > 0) {
					--spin_left;

					bool all_are_stagnant = true;
					// Bost the best species
					real_t luck_bost = 3;
					real_t roulet_ball_pos = Math::randd();

					for (auto it = ordered_species.begin(); it != ordered_species.end(); ++it) {

						if ((*it)->get_stagnant_epochs() <= settings.species_stagnant_age_threshold) {

							all_are_stagnant = false;
							real_t roulet_spot_left = roulet_spot_right;
							roulet_spot_right += roulet_threshold;

							// Make roulet spots circular
							if (roulet_spot_right >= 1) {
								roulet_spot_left = roulet_spot_right;
								roulet_spot_right = roulet_threshold;
							}

							if (roulet_ball_pos > roulet_spot_left &&
									roulet_ball_pos <= roulet_spot_right * luck_bost) {

								(*it)->set_offspring_count(
										(*it)->get_offspring_count() + 1);

								(*it)->set_champion_offspring_count(
										(*it)->get_champion_offspring_count() + 1);

								--stolen_cribs;
								break;
							}
						}

						luck_bost -= 0.3;
						luck_bost = MAX(luck_bost, 1); // Never below 1
					}

					// Premature stop if all are stagnant
					if (all_are_stagnant)
						break;
				}
			}
		}

		// Assign all the remaining stolen cribs to the best species even if it
		// is going to die
		if (stolen_cribs > 0) {

			best_species->set_offspring_count(
					best_species->get_offspring_count() + stolen_cribs);

			best_species->set_champion_offspring_count(
					best_species->get_champion_offspring_count() + stolen_cribs);

			stolen_cribs = 0;
		}

		ERR_FAIL_COND_V(stolen_cribs != 0, false);
	}

	{
		/// Population champion species protection mechanism
		/// Can happen that the pop champion species is stagnant, and go torward
		/// the death.
		/// This must be prevented, so here a gift is donated to make the species
		/// survive.
		/// Also in the next iteration if the species does well will have the
		/// possibility to repopulate.
		NtSpecies *specie_to_save = population_champion->get_species();
		if (specie_to_save->get_offspring_count() == 0) {

			// Search the species to whom steal 1 crib no matter who
			NtSpecies *target = nullptr;
			for (auto it = ordered_species.rbegin(); it != ordered_species.rend(); ++it) {
				NtSpecies *s = *it;
				if (s->get_offspring_count()) {
					target = s;
					break;
				}
			}

			ERR_FAIL_COND_V(!target, false);
			target->set_offspring_count(target->get_offspring_count() - 1);
			if (target->get_champion_offspring_count() >= target->get_champion_offspring_count()) {
				target->set_champion_offspring_count(target->get_champion_offspring_count() - 1);
			}
			specie_to_save->set_offspring_count(1);
			specie_to_save->reset_age_of_last_improvement();
		}
	}

	statistics.species_best_offspring = best_species->get_offspring_count();
	statistics.species_best_champion_offspring = best_species->get_champion_offspring_count();

	/// Step 7. Reproduction phase.
	kill_organisms_marked_for_death();

	// Prepares the organism pool to repopulate
	organisms.clear();
	population_champion = nullptr;

	// Make the fittest organism reproduct
	for (auto it = species.begin(); it != species.end(); ++it) {
		(*it)->reproduce(innovations);
	}

	// Speciate the newest organism
	speciate();

	// Kill older organisms that still inside the species
	for (auto it = species.begin(); it != species.end(); ++it) {
		(*it)->kill_old_organisms();
	}

	// Make rid of void species
	kill_void_species();

	/// Step 8. Population verification

	// Checks if the organisms size is correct
	ERR_FAIL_COND_V(organisms.size() != population_size, false);

	// Checks if the best species still alive
	auto best_species_iterator =
			std::find(species.begin(), species.end(), best_species);
	if (best_species_iterator == species.end()) {
		best_species = nullptr;
		statistics.species_best_is_died = true;
		WARN_PRINTS("IMPORTANT For some reason the best species died. Is there a bug somewhere?");
	}

	bool exist_champion_clone = false;
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		if ((*it)->is_champion_clone()) {
			exist_champion_clone = true;
			break;
		}
	}
	ERR_FAIL_COND_V(!exist_champion_clone, false);

	statistics.is_epoch_advanced = true;

	// Finally Done!
	return true;
}

real_t brain::NtPopulation::get_best_personal_fitness() const {
	return best_personal_fitness;
}

void brain::NtPopulation::get_champion_network(
		brain::SharpBrainArea &r_brain_area) const {

	champion_genome.generate_neural_network(r_brain_area);
}

const brain::NtEpochStatistics &brain::NtPopulation::get_epoch_statistics() const {
	return statistics;
}

void brain::NtPopulation::speciate() {

	for (auto it_o = organisms.begin(); it_o != organisms.end(); ++it_o) {
		NtOrganism *o = *it_o;

		if (o->get_species())
			continue; // This organism is already speciated

		NtSpecies *compatible_species(nullptr);

		// Search compatible specie
		for (auto it_s = species.begin(); it_s != species.end(); ++it_s) {

			NtSpecies *s = *it_s;
			if (!s->size())
				continue;

			NtOrganism *spokesman = s->get_organism(0);
			const real_t compatibility = NtGenetic::compatibility(
					o->get_genome(),
					spokesman->get_genome(),
					settings.genetic_disjoints_significance,
					settings.genetic_excesses_significance,
					settings.genetic_weights_significance);

			if (compatibility <= settings.genetic_compatibility_threshold) {
				compatible_species = s;
				break;
			}
		}

		/// If no specie is available please, create new one
		if (!compatible_species) {
			compatible_species = create_species();
			ERR_FAIL_COND(!compatible_species);
		}

		add_organism_to_species(o, compatible_species);
	}
}

void brain::NtPopulation::kill_void_species() {

	/// Removes all species with 0 organisms
	auto it = species.begin();
	while (it != species.end()) {

		if (!(*it)->size()) {
			it = destroy_species(it);
		} else {
			++it;
		}
	}
}

brain::NtSpecies *brain::NtPopulation::create_species() {
	NtSpecies *new_species = new NtSpecies(this, ++species_last_index, epoch);
	species.push_back(new_species);
	return new_species;
}

void brain::NtPopulation::destroy_species(NtSpecies *p_species) {
	auto it_s = std::find(species.begin(), species.end(), p_species);
	ERR_FAIL_COND(it_s == species.end());
	destroy_species(it_s);
}

std::vector<brain::NtSpecies *>::iterator brain::NtPopulation::destroy_species(
		std::vector<NtSpecies *>::iterator p_species_iterator) {
	NtSpecies *s(*p_species_iterator);
	auto ret = species.erase(p_species_iterator);
	delete s;
	return ret;
}

void brain::NtPopulation::destroy_all_species() {
	for (
			auto it = species.begin();
			it != species.end();
			++it) {

		delete (*it);
		*it = nullptr;
	}
	organisms.clear();
}

brain::NtOrganism *brain::NtPopulation::create_organism() {
	ERR_FAIL_COND_V(organisms.size() >= population_size, nullptr);
	NtOrganism *o = new NtOrganism(this);
	organisms.push_back(o);
	return o;
}

void brain::NtPopulation::destroy_organism(NtOrganism *p_organism) {
	auto iterator = std::find(organisms.begin(), organisms.end(), p_organism);
	ERR_FAIL_COND(iterator == organisms.end());
	destroy_organism(iterator);
}

std::vector<brain::NtOrganism *>::iterator brain::NtPopulation::destroy_organism(
		std::vector<NtOrganism *>::iterator p_organism_iterator) {

	remove_organism_from_species(*p_organism_iterator);
	NtOrganism *o = *p_organism_iterator;
	auto ret = organisms.erase(p_organism_iterator);
	delete o;
	return ret;
}

void brain::NtPopulation::destroy_all_organisms() {
	for (
			auto it = organisms.begin();
			it != organisms.end();
			++it) {

		remove_organism_from_species(*it);
		delete (*it);
	}
	organisms.clear();
}

void brain::NtPopulation::kill_organisms_marked_for_death() {
	auto it = organisms.begin();
	while (it != organisms.end()) {
		if ((*it)->is_marked_for_death())
			it = destroy_organism(it);
		else
			++it;
	}
}

void brain::NtPopulation::add_organism_to_species(
		NtOrganism *p_organism,
		NtSpecies *p_species) {

	ERR_FAIL_COND(p_organism->get_species());
	p_species->add_organism(p_organism);
	p_organism->set_species(p_species);
}

void brain::NtPopulation::remove_organism_from_species(NtOrganism *p_organism) {
	if (!p_organism->get_species())
		return;
	p_organism->get_species()->remove_organism(p_organism);
	p_organism->set_species(nullptr);
}

brain::NtOrganism *brain::NtPopulation::get_rand_champion(
		const NtSpecies *p_except_species) const {

	if (!species.size())
		return nullptr;

	if (p_except_species && species.size() == 1)
		return nullptr;

	const int rand_index =
			static_cast<int>(Math::random(0, species.size() - 1) + 0.5);

	NtSpecies *rand_species(nullptr);
	if (species[rand_index] != p_except_species) {
		rand_species = species[rand_index];
	} else {
		// This is the case when the random index point the exception
		// to fix this, remove or add 1 to the index
		if (rand_index + 1 >= species.size()) {
			// Can't add 1, subtract
			rand_species = species[rand_index - 1];
		} else {
			// Can add 1, add
			rand_species = species[rand_index + 1];
		}
	}

	if (!rand_species)
		return nullptr; // No species available

	return rand_species->get_champion();
}

real_t brain::NtPopulation::rand_gaussian(real_t p_x, void *p_data) {
	NtPopulation *pop = static_cast<NtPopulation *>(p_data);
	return p_x + pop->gaussian_distribution(pop->rand_generator);
}

real_t brain::NtPopulation::rand_cold_gaussian(real_t p_x, void *p_data) {
	NtPopulation *pop = static_cast<NtPopulation *>(p_data);
	return pop->gaussian_distribution(pop->rand_generator);
}
