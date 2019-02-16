#pragma once

#include "brain/math/matrix.h"
#include <vector>

namespace brain {
class BrainArea {

public:
	enum Activation {
		ACTIVATION_SIGMOID,
		ACTIVATION_MAX
	};

	struct LearningCache {
		std::vector<brain::Matrix> layers_output;
	};

private:
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;
	std::vector<Activation> activations;

public:
	BrainArea();
	BrainArea(
			uint32_t p_input_layer_size,
			uint32_t p_hidden_layers_count,
			uint32_t p_output_layer_size);

	void set_input_layer_size(uint32_t p_size);
	uint32_t get_input_layer_size() const;

	void set_output_layer_size(uint32_t p_size);
	uint32_t get_output_layer_size() const;

	void set_hidden_layers_count(uint32_t p_count);
	uint32_t get_hidden_layers_count() const;

	void set_hidden_layer(
			uint32_t p_hidden_layer,
			uint32_t p_size,
			Activation p_activation);

	void set_hidden_layer_size(uint32_t p_hidden_layer, uint32_t p_size);
	uint32_t get_hidden_layer_size(uint32_t p_hidden_layer) const;

	void set_hidden_layer_activation(uint32_t p_layer, Activation p_activation);
	Activation get_hidden_layer_activation(uint32_t p_layer) const;

	void randomize_weights(real_t p_range);
	void fill_weights(real_t p_value);

	void randomize_biases(real_t p_range);
	void fill_biases(real_t p_value);

	int get_layer_count();
	const Matrix &get_layer_weights(const int p_layer) const;

	void set_weight(int p_index, const Matrix &p_matrix);
	void set_biases(int p_index, const Matrix &p_matrix);
	void set_activations(int p_index, Activation p_activation);

	const std::vector<Matrix> &get_weights() { return weights; }
	const std::vector<Matrix> &get_biases() { return biases; }
	const std::vector<Activation> &get_activations() { return activations; }

	/**
	 * @brief learn is a function used to train the brain area. It uses the
	 *	stochastic gradient descent algorithm to adjust weights internally
	 * @param p_guess
	 * @param p_expected
	 * @param p_learn_rate usually something around 0.1
	 * @param p_cache if null the cache is cleared for each call
	 * @return Returns the error of this guess, 0 == Accurate
	 */
	real_t learn(
			const Matrix &p_input,
			const Matrix &p_expected,
			real_t p_learn_rate,
			LearningCache *p_cache);

	/**
	 * @brief guess
	 * @param p_input Input data
	 * @param r_guess result
	 */
	void guess(
			const Matrix &p_input,
			Matrix &r_guess,
			LearningCache *p_cache = nullptr);

	/// Metadata
	/// First is an uint32_t with the size of the entire buffer
	/// Second is an uint32_t that point the size of the real_t
	/// Third is an uint32_t with the weight count
	/// Forth is an uint32_t with the biases count
	/// Fifth is an uint32_t with the activation count
	/// From now on all arrays store in this order weights, biases, activations

	enum MetadataIndices {
		METADATA_BUFFER_SIZE,
		METADATA_REAL_SIZE,
		METADATA_WEIGHT_COUNT,
		METADATA_BIAS_COUNT,
		METADATA_ACTIVATION_COUNT,
		METADATA_MAX
	};

	/**
	 * @brief get_buffer_metadata_size returns the size of metadata
	 * @param p_buffer_metadata
	 * @return
	 */
	int get_buffer_metadata_size() const;

	/**
	 * @brief get_buffer_size Read the metadata and returns the size of the entire buffer
	 * @param p_buffer_metadata
	 * @return
	 */
	uint32_t get_buffer_size(const std::vector<uint8_t> &p_buffer_metadata) const;

	/**
	 * @brief set_buffer give the possibility to restore the knowledge
	 * previously acquired
	 * @param p_buffer
	 * @return
	 */
	bool set_buffer(const std::vector<uint8_t> &p_buffer);

	/**
	 * @brief get_buffer return a buffer with the current knowledge
	 * @param p_buffer
	 * @return
	 */
	bool get_buffer(std::vector<uint8_t> &r_buffer) const;

private:
	void set_layer_size(uint32_t p_layer, uint32_t p_size);
	uint32_t get_layer_size(uint32_t p_layer) const;
};
} // namespace brain
