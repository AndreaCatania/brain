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
		settings(p_settings),
		rand_generator(p_settings.seed),
		gaussian_distribution(0, p_settings.learning_deviation),
		epoch(1),
		best_personal_fitness(0.f),
		epoch_last_improvement(epoch) {

	organisms.resize(p_population_size, nullptr);

	for (
			auto it = organisms.begin();
			it != organisms.end();
			++it) {

		*it = new NtOrganism(this);
		NtGenome &organism_genome = (*it)->get_genome_mutable();
		p_ancestor_genome.duplicate_in(organism_genome);

		organism_genome.map_link_weights(
				rand_gaussian,
				static_cast<void *>(this));
	}

	innovation_number = p_ancestor_genome.get_innovation_number();

	speciate();
}

brain::NtPopulation::~NtPopulation() {
	destroy_all_organism();
	destroy_all_species();
}

uint32_t brain::NtPopulation::get_epoch() const {
	return epoch;
}

uint32_t brain::NtPopulation::get_organism_count() const {
	return organisms.size();
}

const brain::SharpBrainArea *brain::NtPopulation::organism_get_network(uint32_t p_organism_i) const {
	ERR_FAIL_INDEX_V(p_organism_i, organisms.size(), nullptr);
	return &organisms[p_organism_i]->get_brain_area();
}

void brain::NtPopulation::organism_add_fitness(uint32_t p_organism_i, real_t p_fitness) const {
	ERR_FAIL_INDEX(p_organism_i, organisms.size());
	organisms[p_organism_i]->add_middle_fitness(p_fitness);
}

void brain::NtPopulation::epoch_advance() {

	++epoch;

	/// Step 1. Compute organisms fitness
	for (auto it_o = organisms.begin(); it_o != organisms.end(); ++it_o) {
		(*it_o)->compute_final_fitness(settings.fitness_exponent);
		(*it_o)->clear_middle_fitness();
	}

	/// Step 2. Compute species average fitness, then adjust it
	for (auto it_s = species.begin(); it_s != species.end(); ++it_s) {
		(*it_s)->compute_average_fitness();
		(*it_s)->adjust_fitness(
				settings.species_youngness_age_threshold,
				settings.species_youngness_multiplier,
				settings.species_stagnant_age_threshold,
				settings.species_stagnant_multiplier,
				settings.species_survival_ratio);
	}

	/// Stage 3. calculates average population fitness
	real_t total_fitness(0);
	for (auto it = organisms.begin(); it != organisms.end(); ++it) {
		total_fitness += (*it)->get_fitness();
	}
	real_t population_average_fitness = total_fitness / organisms.size();

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
		if (total_expected_offsprings < organisms.size()) {
			int biggest_offsprings = 0;
			NtSpecies *best_species = nullptr;
			for (auto it = species.begin(); it != species.end(); ++it) {
				if (biggest_offsprings < (*it)->get_offspring_count()) {
					biggest_offsprings = (*it)->get_offspring_count();
					best_species = *it;
				}
			}
			best_species->set_offspring_count(biggest_offsprings++);
			++total_expected_offsprings;

			// When the total_expected still below
			// could mean that there is a bug somewhere
			// or the best species is a stagnant species and it get killed by its age
			// What happens is that the population rebord from the best specie
			if (total_expected_offsprings < organisms.size()) {
				WARN_PRINTS("The expected offsprings are below normality, The population is going to rebord from the best population.");
				for (auto it = species.begin(); it != species.end(); ++it) {
					(*it)->set_offspring_count(0);
				}
				best_species->set_offspring_count(organisms.size());
			}
		}
	}

	/// Step 6. Perform offspring assignment
	/// This phase changes depending if the population is stagnant or not
	/// When the pop is not stagnant and is allowed to stole cribs from
	/// the lowest species the stoling process is performed
	std::vector<NtSpecies *> ordered_species(species.size());
	std::copy(species.begin(), species.end(), ordered_species.begin());

	std::sort(
			ordered_species.begin(),
			ordered_species.end(),
			species_fitness_comparator);

	NtSpecies *best_species = *ordered_species.begin();
	NtOrganism *population_champion = best_species->get_champion();
	ERR_FAIL_COND(!population_champion);

	if (best_personal_fitness < population_champion->get_personal_fitness()) {
		best_personal_fitness = population_champion->get_personal_fitness();
		epoch_last_improvement = epoch;
	}

	if (epoch - epoch_last_improvement > settings.population_stagnant_age_thresold) {

		// Stagnant population

		if (ordered_species.size() < 2) {

			WARN_PRINTS("The population is stagnant, and there is only one species. Trying to repopulate, but the expectation is low.");

			best_species->set_offspring_count(organisms.size());
			best_species->set_champion_offspring_count(organisms.size());
			best_species->reset_age_of_last_improvement();

		} else {

			WARN_PRINTS("The population is stagnant, Repopulates from the champions of the best two species.");

			NtSpecies *first_species = *(ordered_species.begin() + 0);
			NtSpecies *second_species = *(ordered_species.begin() + 1);

			const int population_half_size = organisms.size() * 0.5;
			// Avoid precision loss during above division operation
			first_species->set_offspring_count(organisms.size() - population_half_size);
			second_species->set_offspring_count(population_half_size);

			first_species->set_champion_offspring_count(organisms.size() - population_half_size);
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
	} else if (settings.population_cribs_stealing) {

		int stolen_cribs(0);

		// Stole the cribs from the worst species
		// Iterates from the botton and skip the champion species
		for (auto it = ordered_species.end() - 1; it != ordered_species.begin(); --it) {
			NtSpecies *s = *it;
			if (s->get_age() > settings.species_stealing_protection_age_threshold &&
					s->get_offspring_count() > settings.species_stealing_limit) {

				// Steal from this species most possible
				const int possible_booty = s->get_offspring_count() -
										   settings.species_stealing_limit -
										   stolen_cribs;

				const int wanted_booty = settings.population_cribs_stealing -
										 stolen_cribs;

				const int booty = MIN(possible_booty, wanted_booty);

				s->set_offspring_count(s->get_offspring_count() - booty);
				stolen_cribs += booty;

				ERR_FAIL_COND(stolen_cribs > settings.population_cribs_stealing);
				if (stolen_cribs == settings.population_cribs_stealing) {
					break;
				}
			}
		}
	}

	int a = 0;
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
		}

		ERR_FAIL_COND(!compatible_species);

		add_organism_to_specie(o, compatible_species);
	}

	/// Removes all species with 0 organisms
	for (
			auto it = species.begin();
			it != species.end();
			++it) {

		NtSpecies *s = *it;
		if (!s->size()) {
			destroy_species(s);
		}
	}
}

brain::NtSpecies *brain::NtPopulation::create_species() {
	NtSpecies *new_species = new NtSpecies(this, epoch);
	species.push_back(new_species);
	return new_species;
}

void brain::NtPopulation::destroy_species(NtSpecies *p_species) {
	auto it_s = std::find(species.begin(), species.end(), p_species);
	ERR_FAIL_COND(it_s == species.end());
	species.erase(it_s);
	delete p_species;
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

void brain::NtPopulation::destroy_all_organism() {
	for (
			auto it = organisms.begin();
			it != organisms.end();
			++it) {

		remove_organism_from_specie(*it);
		delete (*it);
		*it = nullptr;
	}
	organisms.clear();
}

void brain::NtPopulation::add_organism_to_specie(
		NtOrganism *p_organism,
		NtSpecies *p_species) {

	ERR_FAIL_COND(p_organism->get_species());
	p_species->add_organism(p_organism);
	p_organism->set_species(p_species);
}

void brain::NtPopulation::remove_organism_from_specie(NtOrganism *p_organism) {

	ERR_FAIL_COND(!p_organism->get_species());
	p_organism->get_species()->remove_organism(p_organism);
	p_organism->set_species(nullptr);
}

real_t brain::NtPopulation::rand_gaussian(real_t p_x, void *p_data) {
	NtPopulation *pop = static_cast<NtPopulation *>(p_data);
	return p_x + pop->gaussian_distribution(pop->rand_generator);
}

real_t brain::NtPopulation::rand_cold_gaussian(real_t p_x, void *p_data) {
	NtPopulation *pop = static_cast<NtPopulation *>(p_data);
	return pop->gaussian_distribution(pop->rand_generator);
}
