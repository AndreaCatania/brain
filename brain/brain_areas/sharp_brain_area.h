#pragma once

#include "brain/brain_areas/brain_area.h"
#include <vector>

namespace brain {

typedef uint32_t NeuronId;
class Neuron;

/**
 * @brief The weighted Link between a Neuron and its parents
 */
struct Link {
	Neuron *neuron;
	real_t weight;
};

/**
 * @brief The Neuron struct represent the unit computation of the brain area
 * It could be an input neuron, hidden neuron or an output neuron and
 * never more than one at same time
 */
struct Neuron {

	/**
	 * @brief parents neurons and its connection weight
	 */
	std::vector<Link> parents;

	/**
	 * @brief activation function of this neuron
	 */
	brain::BrainArea::Activation activation;

	/**
	 * @brief The id of neuron, it indicate the index in the neurons vector
	 * of the SharpBrainArea class
	 */
	NeuronId id;

	/**
	 * @brief When the execution_id is the same this value is returned
	 */
	mutable real_t cached_value;

	/**
	 * @brief execution_id Used to know if the cached_value is still valid
	 */
	mutable uint32_t execution_id;

	/**
	 * @brief Neuron constructor
	 * @param p_id
	 */
	Neuron(NeuronId p_id);

	/**
	 * @brief has_parent
	 * @param p_neuron
	 * @return
	 *
	 * Return true if this p_neuron is already a parent of this neuron
	 */
	bool has_parent(Neuron *p_neuron) const;

	/**
	 * @brief get_parent_count
	 * @return
	 */
	uint32_t get_parent_count() const;

	/**
	 * @brief add_parent
	 * @param p_neuron
	 * @param p_weight
	 *
	 * Put the p_neuron as parent and create a weighted link
	 */
	void add_parent(Neuron *p_neuron, real_t p_weight);

	/**
	 * @brief set_weight between this neuron and its parent
	 * @param p_parent_index
	 * @param p_weight
	 */
	void set_weight(NeuronId p_parent_index, real_t p_weight);

	/**
	 * @brief force_set_value
	 * @param p_val
	 * @param p_execution_id
	 *
	 * Used to set the values of the inputs
	 */
	void force_set_value(real_t p_val, uint32_t p_execution_id) const;

	/**
	 * @brief get_value
	 * @param p_execution_id
	 * @return
	 *
	 * Get the value of this neuron, if the passed execution_id is the
	 * same of the stored one the cached value is returned.
	 */
	real_t get_value(uint32_t p_execution_id) const;
};

/**
 * @brief The SharpBrainArea class is the type of brain area that give
 * the possibility to create partially connected neural network.
 *
 * The guess function execution of this neural network is generally 2 times
 * faster than the uniform brain area, on the other hand its creation is slower
 *
 * IMPORTANT: in this brain area the bias is not added automatically, so it
 * must be explicitly created and treated as input neuron.
 */
class SharpBrainArea : public brain::BrainArea {

	/**
	 * @brief execution_id Used to know if the cached_value is still valid
	 */
	mutable uint32_t execution_id;

	/**
	 * @brief neurons of this brain area
	 */
	std::vector<Neuron> neurons;

	/**
	 * @brief inputs neuron ids of this brain area
	 */
	std::vector<NeuronId> inputs;

	/**
	 * @brief outputs neuron ids of this brain area
	 */
	std::vector<NeuronId> outputs;

	/**
	 * @brief ready tells if the network is fully connected
	 * and ready to be used
	 */
	bool ready;

public:
	/**
	 * @brief SharpBrainArea constructor
	 */
	SharpBrainArea();

	/**
	 * @brief add_neuron
	 * @return
	 *
	 * Add an unconnected neuron.
	 * It can become an input, hidden or output neuron
	 */
	NeuronId add_neuron();

	/**
	 * @brief is_neuron_input
	 * @param p_neuron_id
	 * @return
	 *
	 * Returns true if the neuron is already an input
	 */
	bool is_neuron_input(NeuronId p_neuron_id) const;

	/**
	 * @brief set_neuron_as_input
	 * @param p_neuron_id
	 *
	 * Set neuron as input
	 */
	void set_neuron_as_input(NeuronId p_neuron_id);

	/**
	 * @brief is_neuron_output
	 * @param p_neuron_id
	 * @return
	 *
	 * Returns false if the neuron is an output
	 */
	bool is_neuron_output(NeuronId p_neuron_id) const;

	/**
	 * @brief set_neuron_as_output
	 * @param p_neuron_id
	 *
	 * Set the neuron as ouput neuron
	 */
	void set_neuron_as_output(NeuronId p_neuron_id);

	/**
	 * @brief add_link
	 * @param p_neuron_parent_id
	 * @param p_neuron_child_id
	 * @param p_weight
	 *
	 * Add a link between parent node and child node
	 * The direction of the linkage is important
	 */
	void add_link(
			NeuronId p_neuron_parent_id,
			NeuronId p_neuron_child_id,
			real_t p_weight = 0.f);

	/**
	 * @brief clear can be used to delete all neurons
	 */
	void clear();

	/**
	 * @brief randomize_weights randomize the weights
	 * between the passed -range and range
	 *
	 * @param p_range
	 */
	virtual void randomize_weights(real_t p_range);

	/**
	 * @brief fill_weights with the passed value
	 * @param p_weight
	 */
	virtual void fill_weights(real_t p_weight);

	/**
	 * @brief get_input_layer_size
	 * @return
	 *
	 * Get the input layer size
	 */
	virtual uint32_t get_input_layer_size() const;

	/**
	 * @brief get_output_layer_size
	 * @return
	 *
	 * Get the output layer size
	 */
	virtual uint32_t get_output_layer_size() const;

	/**
	 * @brief guess
	 * @param p_input Input data
	 * @param r_guess result
	 *
	 * Make a guess the result considering the inputs
	 *
	 * May fail if the network is not connected
	 */
	virtual void guess(
			const Matrix &p_input,
			Matrix &r_guess) const;

private:
	/**
	 * @brief is_fully_linked_to_input tell to you if the neuron parents are
	 * connected to the input
	 *
	 * @param p_neuron
	 * @return
	 *
	 */
	bool is_fully_linked_to_input(Neuron *p_neuron) const;

	/**
	 * @brief check_if_ready
	 *
	 * This set the ready variable to true only if the network is
	 * correctly connected
	 */
	void check_if_ready();

	/**
	 * @brief randomize_parents_weight is used to randomize the weight between
	 * the parents and this neuron between a range of -p_range and p_range
	 *
	 * @param p_neuron
	 * @param p_range
	 */
	void randomize_parents_weight(Neuron *p_neuron, real_t p_range);

	/**
	 * @brief set_parents_weight is used to set the weights to all parents of
	 * this neuron
	 *
	 * @param p_neuron
	 * @param p_weight
	 */
	void set_parents_weight(Neuron *p_neuron, real_t p_weight);
};

} // namespace brain
