#include "neat_population.h"

#include "brain/math/math_funcs.h"

brain::NtPopulation::NtPopulation(
		const NtGenome &p_ancestor_genome,
		int p_population_size,
		real_t p_splitting_threshold,
		uint64_t p_seed,
		real_t p_learning_deviation) :
		splitting_threshold(p_splitting_threshold),
		rand_generator(p_seed),
		gaussian_distribution(0, p_learning_deviation) {

	organisms.resize(p_population_size, this);

	for (
			auto it = organisms.begin();
			it != organisms.end();
			++it) {

		NtGenome &organism_genome = it->get_genome_mutable();
		p_ancestor_genome.duplicate_in(organism_genome);

		organism_genome.map_link_weights(
				rand_gaussian,
				static_cast<void *>(this));
	}

	innovation_number = p_ancestor_genome.get_innovation_number();

	speciate();
}

void brain::NtPopulation::speciate() {
}

real_t brain::NtPopulation::rand_gaussian(real_t p_x, void *p_data) {
	NtPopulation *pop = static_cast<NtPopulation *>(p_data);
	return p_x + pop->gaussian_distribution(pop->rand_generator);
}

real_t brain::NtPopulation::rand_cold_gaussian(real_t p_x, void *p_data) {
	NtPopulation *pop = static_cast<NtPopulation *>(p_data);
	return pop->gaussian_distribution(pop->rand_generator);
}
