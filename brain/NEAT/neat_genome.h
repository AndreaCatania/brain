#pragma once

#include "brain/brain_areas/sharp_brain_area.h"
#include <vector>

typedef real_t (*map_real_1)(real_t p_arg_1);
typedef real_t (*map_real_2_ptr)(real_t p_arg_1, void *p_data);

namespace brain {

/**
 * @brief The NtGene struct is the base type of each gene
 */
struct NtGene {
};

/**
 * @brief The NtNeuronGene struct stores all information abount neurons
 */
struct NtNeuronGene : public NtGene {

	/**
	 * @brief The NeuronGeneType enum indicate the type of the neuron gene
	 */
	enum NeuronGeneType {
		NEURON_GENE_TYPE_INPUT,
		NEURON_GENE_TYPE_HIDDEN,
		NEURON_GENE_TYPE_OUTPUT
	};

	/**
	 * @brief NtNeuronGene constructor
	 * @param p_id
	 * @param p_type
	 */
	NtNeuronGene(uint32_t p_id, NeuronGeneType p_type);

	/**
	 * @brief id of the neuron, it's also the index in the neuron_genes vector
	 */
	uint32_t id;

	/**
	 * @brief type is used to store the type of the neuron gene
	 */
	NeuronGeneType type;

	/**
	 * @brief incoming_links is the vector with the ID of the incoming links
	 */
	std::vector<int> incoming_links;

	/**
	 * @brief outcoming_links is the vector with the ID of the outcoming links
	 */
	std::vector<int> outcoming_links;
};

/**
 * @brief The NtLinkGene struct is used to correctly connect two neurons
 * Also the link can be deactivated.
 */
struct NtLinkGene : public NtGene {

	/**
	 * @brief Void constructor
	 */
	NtLinkGene();

	/**
	 * @brief LinkGene constructor
	 * @param p_id
	 * @param p_active
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @param p_weight
	 * @param p_recurrent
	 * @param p_innovation_number
	 */
	NtLinkGene(
			uint32_t p_id,
			bool p_active,
			uint32_t p_parent_neuron_id,
			uint32_t p_child_neuron_id,
			real_t p_weight,
			bool p_recurrent,
			uint32_t p_innovation_number);

	/**
	 * @brief self id of the link, it's also the index in the link_genes vector
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
	 * @brief When recurrent is true the connection go in the opposite direction
	 * and the returned value is the one calculated in the previous iteration
	 */
	bool recurrent;

	/**
	 * @brief innovation_number is used to maintain the history of the
	 * Genome evolution
	 */
	uint32_t innovation_number;
};

/**
 * @brief The Innovation struct is used to track all innovation that happens
 * during the innovation phase.
 * This is necessary in order to assign the correct innovation number to a
 * mutation.
 */
struct NtInnovation {

	/**
	 * @brief The InnovationType enum
	 */
	enum InnovationType {
		INNOVATION_NODE,
		INNOVATION_LINK
	};

	/**
	 * @brief The type of the innovation
	 */
	InnovationType type;

	/**
	 * @brief The id of the parent neuron
	 */
	uint32_t parent_neuron_id;

	/**
	 * @brief The id of the child neuron
	 */
	uint32_t child_neuron_id;

	/**
	 * @brief is recurrent link
	 */
	bool is_recurrent;

	/**
	 * @brief The innovation number of this innovation
	 */
	uint32_t innovation_number;

	/**
	 * @brief neuron_id used only with INNOVATION_NODE
	 */
	uint32_t neuron_id;
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
	std::vector<NtNeuronGene> neuron_genes;

	/**
	 * @brief link_genes stores all linkage information used to create the
	 * Neural Network structure
	 */
	std::vector<NtLinkGene> link_genes;

	/**
	 * @brief biggest_innovation_number
	 */
	uint32_t biggest_innovation_number;

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
			NtNeuronGene::NeuronGeneType p_type);

	/**
	 * @brief add_link add an active link gene between two neuron to the genome
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @param p_weight
	 * @param p_recurrent
	 * @param p_innovation_number
	 * @return the id of the added link
	 */
	uint32_t add_link(
			uint32_t p_parent_neuron_id,
			uint32_t p_child_neuron_id,
			real_t p_weight,
			bool p_recurrent,
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
	const NtLinkGene *get_link(int p_i) const;

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
	 * @brief mutate_all_link_weights map all weights
	 * @param p_map_func
	 */
	void mutate_all_link_weights(map_real_1 p_map_func);

	/**
	 * @brief mutate_all_link_weights map all weights
	 * @param p_map_func
	 * @param p_data is passed directly to the map_func
	 */
	void mutate_all_link_weights(map_real_2_ptr p_map_func, void *p_data);

	/**
	 * @brief Mutates the link of just one random weight
	 * @param p_map_func
	 * @param p_data
	 */
	void mutate_random_link_weight(map_real_2_ptr p_map_func, void *p_data);

	/**
	 * @brief mutate_random_link_toggle_activation take a random link and toggle its
	 * activation status
	 */
	void mutate_random_link_toggle_activation();

	/**
	 * @brief add a random link between nodes, depending on the spwn recurrent
	 * threshold is possible to spawn a recurrent link
	 * @param p_spawn_recurrent_threshold
	 * @param r_innovations Shared Innovation list that is updates in case of new innovation
	 * @param r_current_innovation_number shared innovation number that is updated in acse of new innovation
	 * @return returns true if the genome is mutated
	 */
	bool mutate_add_random_link(
			real_t p_spawn_recurrent_threshold,
			std::vector<NtInnovation> &r_innovations,
			uint32_t &r_current_innovation_number);

	/**
	 * @brief mutate_add_random_neuron will add a neuron in between two neurons,
	 * the link that connect them get broken, and another two get born to connect
	 * this new neuron
	 * @param r_innovations
	 * @param r_current_innovation_number
	 * @return
	 */
	bool mutate_add_random_neuron(
			std::vector<NtInnovation> &r_innovations,
			uint32_t &r_current_innovation_number);

	/// Cross over operations ---V
	/// https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
	/**
	 * @brief mate_multipoint will delete all genes of the current genome
	 * and will spawn new genes from the mating between mom and daddy genomes.
	 *
	 * The multipoint mating, will choose randomly from one or another parent
	 * all the genes with the same innovation number, and will choose to pick
	 * or refiuse a gene that is present only to a certain parent depending on
	 * the fitness and a probability threshold.
	 *
	 * When the p_average is true instead to choose one or the other the weights
	 * of the genes with the same innovation number get averaged.
	 *
	 * @param p_mom
	 * @param p_mom_fitness
	 * @param p_daddy
	 * @param p_daddy_fitness
	 * @param p_average
	 * @return
	 */
	bool mate_multipoint(
			const NtGenome &p_mom,
			real_t p_mom_fitness,
			const NtGenome &p_daddy,
			real_t p_daddy_fitness,
			bool p_average);

	/**
	 * @brief mate_singlepoint will choose a random point inside the smaller
	 * genome and will perform a cut in both genomes then
	 * they will crossed.
	 * The cutted gene instead get averaged.
	 *
	 * @param p_mom
	 * @param p_daddy
	 * @return
	 */
	bool mate_singlepoint(
			const NtGenome &p_mom,
			const NtGenome &p_daddy);

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
	 * @brief sort_genes using the innovation number
	 */
	void sort_genes();

	/**
	 * @brief check_innovation_numbers returns true if the genes are ordered
	 * by innovation number
	 * @return
	 */
	bool check_innovation_numbers() const;

	/**
	 * @brief get_innovation_number returns the biggest innovation number
	 * @return
	 */
	uint32_t get_innovation_number() const;

	/**
	 * @brief is_link_recurrent is used to know if a link should be
	 * recurrent or not, this can be used even before the real existence of the
	 * link.
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @return
	 */
	bool is_link_recurrent(
			NeuronId p_parent_neuron_id,
			NeuronId p_child_neuron_id) const;

private:
	/**
	 * @brief _recursive_is_link_recurrent is a function that really operate the
	 * is_link_recurrent work.
	 *
	 * This function require some strange inputs, from the prospective of user,
	 * for this reason is wrapped by the is_link_recurrent.
	 *
	 * NOTE: The scope of this function is to make sure to not get stuck if
	 * there is a loop in the structure.
	 *
	 * @param p_parent_neuron_id
	 * @param p_middle_neuron_id
	 * @param p_child_neuron_id
	 * @return
	 */
	bool _recursive_is_link_recurrent(
			NeuronId p_parent_neuron_id,
			NeuronId p_middle_neuron_id,
			NeuronId p_child_neuron_id) const;

	/** TODO please handle this properly
	 * @brief find_innovation is an utility function that search the innovation
	 * inside the passed array and return the index to the innovation or -1
	 * @param p_innovations
	 * @param p_innovation_type
	 * @param p_parent_neuron_id
	 * @param p_child_neuron_id
	 * @param p_is_recurrent
	 * @param p_neuron_id
	 * @return
	 */
	static int find_innovation(
			std::vector<NtInnovation> &p_innovations,
			NtInnovation::InnovationType p_innovation_type,
			NeuronId p_parent_neuron_id,
			NeuronId p_child_neuron_id,
			bool p_is_recurrent,
			uint32_t p_neuron_id);
};

} // namespace brain

extern bool gene_innovation_comparator(brain::NtLinkGene &p_1, brain::NtLinkGene &p_2);
