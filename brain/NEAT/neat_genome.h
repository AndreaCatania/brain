#pragma once

#include "brain/brain_areas/sharp_brain_area.h"
#include <vector>

namespace brain {

/**
 * @brief The Gene struct is the base type of each gene
 */
struct Gene {
};

/**
 * @brief The NeuronGene struct stores all information abount neurons
 */
struct NeuronGene : public Gene {

	/**
	 * @brief The NeuronGeneType enum indicate the type of the neuron gene
	 */
	enum NeuronGeneType {
		NEURON_GENE_TYPE_INPUT,
		NEURON_GENE_TYPE_HIDDEN,
		NEURON_GENE_TYPE_OUTPUT
	};

	/**
	 * @brief id of the neuron, it's also the index in the neuron_genes vector
	 */
	uint32_t id;

	/**
	 * @brief type is used to store the type of the neuron gene
	 */
	NeuronGeneType type;
};

/**
 * @brief The LinkGene struct is used to correctly connect two neurons
 * Also the link can be deactivated.
 */
struct LinkGene : public Gene {

	/**
	 * @brief active is used to active and deactivate the link.
	 * This is used to make possible keep the history of genes in the Genome
	 */
	bool active;

	/**
	 * @brief parent_neuron_id store the neuron id of the parent,
	 * in this connection
	 */
	uint32_t parent_neuron_id;

	/**
	 * @brief child_neuron_id store the neuron id of the child,
	 * in this connection
	 */
	uint32_t child_neuron_id;

	/**
	 * @brief weight of the connection
	 */
	real_t weight;

	/**
	 * @brief innovation_number is used to maintain the history of the
	 * Genome evolution
	 */
	uint32_t innovation_number;
};

/**
 * @brief The NEATGenome class is the organism structure description that can
 * be used to generates the phenotype that is neural network
 *
 * To maintain the history of the mutations this class keep all genes but
 * deactivates the mutated ones.
 */
class NEATGenome {

	/**
	 * @brief neuron_genes store all neuron information used to create the
	 * phenotype.
	 * The order is important
	 */
	std::vector<NeuronGene> neuron_genes;

	/**
	 * @brief link_genes stores all linkage information used to create the
	 * Neural Network structure
	 */
	std::vector<LinkGene> link_genes;

public:
	/**
	 * @brief NEATGenome constructor
	 */
	NEATGenome();

	/**
	 * @brief generate_neural_network is used to generate the phenotype using
	 * the description of this Genome.
	 *
	 * @param r_brain_area
	 */
	void generate_neural_network(SharpBrainArea &r_brain_area) const;
};

} // namespace brain
