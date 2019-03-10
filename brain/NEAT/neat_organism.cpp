#include "neat_organism.h"

#include "brain/NEAT/neat_population.h"
#include "brain/NEAT/neat_species.h"
#include "brain/error_macros.h"

brain::NtOrganism::NtOrganism(const NtPopulation *p_owner) :
		owner(p_owner),
		species(nullptr),
		dead(true),
		is_dirty_brain_area(true) {
}

brain::NtOrganism::~NtOrganism() {
	if (species) {
		ERR_PRINTS("The organism belongs to a species, remove this before destruct the organism");
	}
}

brain::NtGenome &brain::NtOrganism::get_genome_mutable() {
	is_dirty_brain_area = true;
	return genome;
}

const brain::NtGenome &brain::NtOrganism::get_genome() const {
	return genome;
}

void brain::NtOrganism::set_species(NtSpecies *p_species) {
	species = p_species;
}

brain::NtSpecies *brain::NtOrganism::get_species() const {
	return species;
}
