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
		genetic_compatibility_threshold(p_settings.genetic_compatibility_threshold),
		genetic_disjoints_significance(p_settings.genetic_disjoints_significance),
		genetic_excesses_significance(p_settings.genetic_excesses_significance),
		genetic_weights_significance(p_settings.genetic_weights_significance),
		rand_generator(p_settings.seed),
		gaussian_distribution(0, p_settings.learning_deviation) {

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
					genetic_disjoints_significance,
					genetic_excesses_significance,
					genetic_weights_significance);

			if (compatibility <= genetic_compatibility_threshold) {
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
	NtSpecies *new_species = new NtSpecies(this);
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
