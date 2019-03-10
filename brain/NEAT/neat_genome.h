#pragma once

#include "brain/brain_areas/sharp_brain_area.h"
#include <vector>

typedef real_t (*map_real_1)(real_t p_arg_1);
typedef real_t (*map_real_2_ptr)(real_t p_arg_1, void *p_data);

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
class NtGenome {

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
	NtGenome();

	/**
	 * @brief NtGenome this constructor created automatically a fully connected
	 * genome with input and outputs passes as parameters.
	 *
	 * This constructor is perfect to create the ancestor genome to pass to
	 * create the population
	 *
	 * @param p_input_count
	 * @param p_output_count
	 * @param p_randomize_weights = true
	 */
	NtGenome(int p_input_count, int p_output_count, bool p_randomize_weights = true);

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
	 * @brief get_link_size get the link count
	 * @return
	 */
	uint32_t get_link_count() const;

	/**
	 * @brief get_link returns the link gene or nullptr if p_i is not valid
	 * @param p_i
	 * @return
	 */
	const LinkGene *get_link(int p_i) const;

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
	 * @brief map_link_weights map the weights
	 * @param p_map_func
	 */
	void map_link_weights(map_real_1 p_map_func);

	/**
	 * @brief map_link_weights map the weights
	 * @param p_map_func
	 * @param p_data is passed directly to the map_func
	 */
	void map_link_weights(map_real_2_ptr p_map_func, void *p_data);

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
	void duplicate_in(NtGenome &p_genome) const;

	/**
	 * @brief get_innovation_number returns the innovation number of the last gene
	 * @return
	 */
	uint32_t get_innovation_number() const;
};

} // namespace brain
