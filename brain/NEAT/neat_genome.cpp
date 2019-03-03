#include "neat_genome.h"

#include "brain/error_macros.h"

brain::NEATGenome::NEATGenome() {

	// TODO remove it, this is just a test

	neuron_genes.push_back({ 0, NeuronGene::NEURON_GENE_TYPE_INPUT });
	//neuron_genes.push_back({ 1, NeuronGene::NEURON_GENE_TYPE_INPUT });
	//neuron_genes.push_back({ 2, NeuronGene::NEURON_GENE_TYPE_HIDDEN });
	//neuron_genes.push_back({ 3, NeuronGene::NEURON_GENE_TYPE_OUTPUT });
	//
	//link_genes.push_back({ true, 0, 2, 1.0, 0 });
	//link_genes.push_back({ true, 0, 3, 1.0, 1 });
	//link_genes.push_back({ true, 1, 2, 1.0, 2 });
	//link_genes.push_back({ true, 2, 3, 1.0, 3 });
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
