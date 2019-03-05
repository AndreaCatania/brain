#include "neat_genome.h"

#include "brain/error_macros.h"

brain::NeuronGene::NeuronGene(uint32_t p_id, NeuronGeneType p_type) :
		id(p_id),
		type(p_type) {}

brain::LinkGene::LinkGene(
		uint32_t p_id,
		bool p_active,
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id,
		real_t p_weight,
		uint32_t p_innovation_number) :
		id(p_id),
		active(p_active),
		parent_neuron_id(p_parent_neuron_id),
		child_neuron_id(p_child_neuron_id),
		weight(p_weight),
		innovation_number(p_innovation_number) {}

brain::NEATGenome::NEATGenome() {}

uint32_t brain::NEATGenome::add_neuron(
		NeuronGene::NeuronGeneType p_type) {

	const uint32_t id(neuron_genes.size());

	neuron_genes.push_back(NeuronGene(
			id,
			p_type));
	return id;
}

uint32_t brain::NEATGenome::add_link(
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id,
		real_t p_weight,
		uint32_t p_innovation_number) {

	DEBUG_ONLY(ERR_FAIL_COND_V(0 > find_link(p_parent_neuron_id, p_child_neuron_id), -1));

	const uint32_t id(link_genes.size());

	link_genes.push_back(
			LinkGene(
					id,
					true,
					p_parent_neuron_id,
					p_child_neuron_id,
					p_weight,
					p_innovation_number));

	return id;
}

void brain::NEATGenome::active_link(uint32_t p_link_id) {
	ERR_FAIL_INDEX(p_link_id, link_genes.size());

	link_genes[p_link_id].active = true;
}

void brain::NEATGenome::suppress_link(uint32_t p_link_id) {
	ERR_FAIL_INDEX(p_link_id, link_genes.size());

	link_genes[p_link_id].active = false;
}

uint32_t brain::NEATGenome::find_link(
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id) {

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		if (p_parent_neuron_id == it->parent_neuron_id)
			if (p_child_neuron_id == it->child_neuron_id)
				return it->id;
	}

	return -1;
}

void brain::NEATGenome::generate_neural_network(SharpBrainArea &r_brain_area) const {

	r_brain_area.clear();

	for (auto it = neuron_genes.begin(); it != neuron_genes.end(); ++it) {

		const uint32_t id = r_brain_area.add_neuron();

		ERR_FAIL_COND(id != it->id);

		switch (it->type) {
			case NeuronGene::NEURON_GENE_TYPE_INPUT:
				r_brain_area.set_neuron_as_input(id);
				break;
			case NeuronGene::NEURON_GENE_TYPE_OUTPUT:
				r_brain_area.set_neuron_as_output(id);
				break;
		}
	}

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		if (!it->active)
			continue;

		r_brain_area.add_link(
				it->parent_neuron_id,
				it->child_neuron_id,
				it->weight);
	}
}

void brain::NEATGenome::clear() {
	neuron_genes.clear();
	link_genes.clear();
}

void brain::NEATGenome::duplicate_in(NEATGenome &p_genome) const {

	p_genome.clear();

	p_genome.neuron_genes.reserve(neuron_genes.size());
	p_genome.link_genes.reserve(link_genes.size());

	// TODO try to use std::copy instead

	for (auto it = neuron_genes.begin(); it != neuron_genes.end(); ++it) {

		p_genome.neuron_genes.push_back(*it);
	}

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {

		p_genome.link_genes.push_back(*it);
	}
}
