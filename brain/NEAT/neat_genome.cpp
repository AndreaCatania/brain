#include "neat_genome.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

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

brain::NtGenome::NtGenome() {}

brain::NtGenome::NtGenome(
		int p_input_count,
		int p_output_count,
		bool p_randomize_weights) :
		NtGenome() {

	ERR_FAIL_COND(p_input_count <= 0);
	ERR_FAIL_COND(p_output_count <= 0);

	for (int i(0); i < p_input_count; ++i) {
		add_neuron(NeuronGene::NEURON_GENE_TYPE_INPUT);
	}

	for (int i(0); i < p_output_count; ++i) {
		add_neuron(NeuronGene::NEURON_GENE_TYPE_OUTPUT);
	}

	int innovation_number(0);
	for (int o_i(0); o_i < p_output_count; ++o_i) {
		for (int i_i(0); i_i < p_input_count; ++i_i) {

			add_link(
					i_i,
					p_input_count + o_i, // Output neurons are after inputs neurons
					p_randomize_weights ? Math::random(-1, 1) : 1,
					innovation_number++);
		}
	}
}

uint32_t brain::NtGenome::add_neuron(
		NeuronGene::NeuronGeneType p_type) {

	const uint32_t id(neuron_genes.size());

	neuron_genes.push_back(NeuronGene(
			id,
			p_type));
	return id;
}

uint32_t brain::NtGenome::add_link(
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

uint32_t brain::NtGenome::get_link_count() const {
	return link_genes.size();
}

const brain::LinkGene *brain::NtGenome::get_link(int p_i) const {
	ERR_FAIL_INDEX_V(p_i, link_genes.size(), nullptr);
	return link_genes.data() + p_i;
}

void brain::NtGenome::active_link(uint32_t p_link_id) {
	ERR_FAIL_INDEX(p_link_id, link_genes.size());

	link_genes[p_link_id].active = true;
}

void brain::NtGenome::suppress_link(uint32_t p_link_id) {
	ERR_FAIL_INDEX(p_link_id, link_genes.size());

	link_genes[p_link_id].active = false;
}

uint32_t brain::NtGenome::find_link(
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id) {

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		if (p_parent_neuron_id == it->parent_neuron_id)
			if (p_child_neuron_id == it->child_neuron_id)
				return it->id;
	}

	return -1;
}

void brain::NtGenome::map_link_weights(map_real_1 p_map_func) {
	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		it->weight = p_map_func(it->weight);
	}
}

void brain::NtGenome::map_link_weights(map_real_2_ptr p_map_func, void *p_data) {
	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		it->weight = p_map_func(it->weight, p_data);
	}
}

bool brain::NtGenome::add_random_link(
		real_t p_spawn_recurrent_threshold,
		std::vector<Innovation> &r_innovations,
		uint32_t &r_current_innovation_number) {

	ERR_FAIL_V(false);
	return false;
}

void brain::NtGenome::generate_neural_network(SharpBrainArea &r_brain_area) const {

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

void brain::NtGenome::clear() {
	neuron_genes.clear();
	link_genes.clear();
}

void brain::NtGenome::duplicate_in(NtGenome &p_genome) const {

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

uint32_t brain::NtGenome::get_innovation_number() const {
	const int last_gene = link_genes.size() - 1;
	ERR_FAIL_INDEX_V(last_gene, link_genes.size(), -1);
	return link_genes[last_gene].innovation_number;
}
