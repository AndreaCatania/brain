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
		std::vector<brain::Matrix> layers_inputs;
		std::vector<brain::Matrix> layers_output;
	};

private:
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;
	std::vector<Activation> activations;

public:
	BrainArea();

	void set_input_layer_size(uint32_t p_size);
	uint32_t get_input_layer_size() const;

	void set_output_layer_size(uint32_t p_size);
	uint32_t get_output_layer_size() const;

	void resize_hidden_layers(uint32_t p_count);
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
	void set_weights(real_t p_value);

	void randomize_biases(real_t p_range);
	void set_biases(real_t p_value);

	int get_layer_count();
	const Matrix &get_layer_weights(const int p_layer) const;

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

private:
	void set_layer_size(uint32_t p_layer, uint32_t p_size);
	uint32_t get_layer_size(uint32_t p_layer) const;
};
} // namespace brain
