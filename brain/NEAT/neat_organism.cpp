#include "neat_organism.h"

#include "brain/NEAT/neat_population.h"

brain::NtOrganism::NtOrganism(const NtPopulation *p_owner) :
		owner(p_owner),
		dead(true),
		is_dirty_brain_area(true) {
}

brain::NtGenome &brain::NtOrganism::get_genome_mutable() {
	is_dirty_brain_area = true;
	return genome;
}
