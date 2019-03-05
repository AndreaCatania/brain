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
	 * @brief NeuronGene constructor
	 * @param p_id
	 * @param p_type
	 */
	NeuronGene(uint32_t p_id, NeuronGeneType p_type);

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
	 * @brief LinkGene constructor
	 * @param p_id
	 * @param p_active
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @param p_weight
	 * @param p_innovation_number
	 */
	LinkGene(
			uint32_t p_id,
			bool p_active,
			uint32_t p_parent_neuron_id,
			uint32_t p_child_neuron_id,
			real_t p_weight,
			uint32_t p_innovation_number);

	/**
	 * @brief id of the link, it's also the index in the link_genes vector
	 */
	uint32_t id;

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
	 * @brief add_neuron add a neuron gene to the genome
	 * @param p_type
	 * @return the id of the added neuron
	 */
	uint32_t add_neuron(
			NeuronGene::NeuronGeneType p_type);

	/**
	 * @brief add_link add an active link gene between two neuron to the genome
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @param p_weight
	 * @param p_innovation_number
	 * @return the id of the added link
	 */
	uint32_t add_link(
			uint32_t p_parent_neuron_id,
			uint32_t p_child_neuron_id,
			real_t p_weight,
			uint32_t p_innovation_number);

	/**
	 * @brief active_link active a link
	 * @param p_link_id
	 */
	void active_link(uint32_t p_link_id);

	/**
	 * @brief suppress_link deactives the link
	 * @param p_link_id
	 */
	void suppress_link(uint32_t p_link_id);

	/**
	 * @brief find_link find the link if exist
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @return -1 if the link doesn't exists otherwise the link id
	 */
	uint32_t find_link(
			uint32_t p_parent_neuron_id,
			uint32_t p_child_neuron_id);

	/**
	 * @brief generate_neural_network is used to generate the phenotype using
	 * the description of this Genome.
	 *
	 * @param r_brain_area
	 */
	void generate_neural_network(SharpBrainArea &r_brain_area) const;

	/**
	 * @brief clear function
	 */
	void clear();

	/**
	 * @brief duplicate_in this genome
	 * @param p_genome
	 */
	void duplicate_in(NEATGenome &p_genome) const;
};

} // namespace brain
