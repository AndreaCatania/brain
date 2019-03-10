#include "neat_species.h"

#include "brain/NEAT/neat_organism.h"
#include "brain/NEAT/neat_population.h"
#include "brain/error_macros.h"
#include <algorithm>

brain::NtSpecies::NtSpecies(const NtPopulation *p_owner) :
		owner(p_owner) {
}

brain::NtSpecies::~NtSpecies() {
	ERR_FAIL_COND(organisms.size());
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
