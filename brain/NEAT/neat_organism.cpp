#include "neat_organism.h"

#include "brain/NEAT/neat_population.h"
#include "brain/NEAT/neat_species.h"
#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

brain::NtOrganism::NtOrganism(const NtPopulation *p_owner, const uint32_t p_id) :
		owner(p_owner),
		id(p_id),
		species(nullptr),
		marked_for_death(false),
		is_dirty_brain_area(true),
		middle_fitness_sum(0.f),
		middle_fitness_count(0),
		fitness(0.f),
		personal_fitness(0.f),
		expected_offspring(0.f),
		log(""),
		brain_area(this) {
}

brain::NtOrganism::~NtOrganism() {
	if (species) {
		ERR_PRINTS("The organism belongs to a species, remove this before destruct the organism");
	}
}

uint32_t brain::NtOrganism::get_id() const {
	return id;
}

brain::NtGenome &brain::NtOrganism::get_genome_mutable() {
	is_dirty_brain_area = true;
	return genome;
}

const brain::NtGenome &brain::NtOrganism::get_genome() const {
	return genome;
}

const brain::SharpBrainArea &brain::NtOrganism::get_brain_area() const {
	if (is_dirty_brain_area) {
		is_dirty_brain_area = false;
		genome.generate_neural_network(brain_area);
	}
	return brain_area;
}

void brain::NtOrganism::set_mark_for_death(bool p_mark) {
	marked_for_death = p_mark;
}

bool brain::NtOrganism::is_marked_for_death() const {
	return marked_for_death;
}

void brain::NtOrganism::set_species(NtSpecies *p_species) {
	species = p_species;
}

brain::NtSpecies *brain::NtOrganism::get_species() const {
	return species;
}

void brain::NtOrganism::set_evaluation(real_t p_fitness) {
	personal_fitness = MAX(p_fitness, CMP_EPSILON);
	fitness = personal_fitness;
}

void brain::NtOrganism::set_fitness(real_t p_fitness) {
	fitness = p_fitness;
}

real_t brain::NtOrganism::get_fitness() const {
	return fitness;
}

real_t brain::NtOrganism::get_personal_fitness() const {
	return personal_fitness;
}

void brain::NtOrganism::set_expected_offspring(real_t p_offspring) {
	expected_offspring = p_offspring;
}

real_t brain::NtOrganism::get_expected_offspring() const {
	return expected_offspring;
}

bool organism_fitness_comparator(brain::NtOrganism *p_1, brain::NtOrganism *p_2) {
	return p_1->get_fitness() > p_2->get_fitness();
}
