#include "brain_area.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

#define HIDDEN_LAYER(layer) (layer + 1)
#define HIDDEN_LAYER_COUNT(count) (count - 2)

#define INPUT_LAYER_ID 0
#define OUTPUT_LAYER_ID weights.size()

#define ACTIVATION_ID(layer) ((layer)-1)
#define WEIGHT_ID(layer) (layer)
#define BIAS_ID(layer) (layer)

matrix_map activation_functions[] = { brain::Math::sigmoid };
matrix_map activation_derivatives[] = { brain::Math::sigmoid_fast_derivative };

brain::BrainArea::BrainArea() {
	weights.resize(1);
	biases.resize(1);
	activations.push_back(ACTIVATION_SIGMOID);
}

brain::BrainArea::BrainArea(
		uint32_t p_input_layer_size,
		uint32_t p_hidden_layers_count,
		uint32_t p_output_layer_size) :
		BrainArea() {
	set_input_layer_size(p_input_layer_size);
	set_hidden_layers_count(p_hidden_layers_count);
	set_output_layer_size(p_output_layer_size);
}

void brain::BrainArea::set_input_layer_size(uint32_t p_size) {
	set_layer_size(INPUT_LAYER_ID, p_size);
}

uint32_t brain::BrainArea::get_input_layer_size() const {
	return get_layer_size(INPUT_LAYER_ID);
}

void brain::BrainArea::set_output_layer_size(uint32_t p_size) {

	set_layer_size(OUTPUT_LAYER_ID, p_size);
}

uint32_t brain::BrainArea::get_output_layer_size() const {
	return get_layer_size(OUTPUT_LAYER_ID);
}

void brain::BrainArea::set_hidden_layers_count(uint32_t p_count) {

	const int prev_size_output_layer = get_layer_size(OUTPUT_LAYER_ID);
	const Activation prev_activ_output_layer = activations[activations.size() - 1];

	weights.resize(p_count + 2 - 1);
	biases.resize(p_count + 2 - 1);
	activations.resize(p_count + 2 - 1);

	set_layer_size(OUTPUT_LAYER_ID, prev_size_output_layer);
	activations[activations.size() - 1] = prev_activ_output_layer;
}

uint32_t brain::BrainArea::get_hidden_layers_count() const {
	return weights.size() - 1;
}

void brain::BrainArea::set_hidden_layer(
		uint32_t p_hidden_layer,
		uint32_t p_size,
		Activation p_activation) {

	set_hidden_layer_size(p_hidden_layer, p_size);
	set_hidden_layer_activation(p_hidden_layer, p_activation);
}

void brain::BrainArea::set_hidden_layer_size(uint32_t p_hidden_layer, uint32_t p_size) {
	set_layer_size(HIDDEN_LAYER(p_hidden_layer), p_size);
}

uint32_t brain::BrainArea::get_hidden_layer_size(uint32_t p_hidden_layer) const {
	return get_layer_size(HIDDEN_LAYER(p_hidden_layer));
}

void brain::BrainArea::set_hidden_layer_activation(uint32_t p_hidden_layer, Activation p_activation) {
	ERR_FAIL_INDEX(p_hidden_layer, activations.size());
	activations[p_hidden_layer] = p_activation;
}

brain::BrainArea::Activation brain::BrainArea::get_hidden_layer_activation(
		uint32_t p_hidden_layer) const {
	ERR_FAIL_INDEX_V(p_hidden_layer, activations.size(), ACTIVATION_MAX);
	return activations[p_hidden_layer];
}

real_t matrix_rand(real_t x, real_t p_range) {
	return brain::Math::random(-p_range, p_range);
}

void brain::BrainArea::randomize_weights(real_t p_range) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].map(matrix_rand, p_range);
	}
}

void brain::BrainArea::fill_weights(real_t p_value) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].set_all(p_value);
	}
}

void brain::BrainArea::randomize_biases(real_t p_range) {

	for (int i(0); i < biases.size(); ++i) {
		biases[i].map(matrix_rand, p_range);
	}
}

void brain::BrainArea::fill_biases(real_t p_value) {
	for (int i(0); i < biases.size(); ++i) {
		biases[i].set_all(p_value);
	}
}

int brain::BrainArea::get_layer_count() {
	return weights.size() + 1;
}

const brain::Matrix &brain::BrainArea::get_layer_weights(const int p_layer) const {
	return weights[p_layer];
}

void brain::BrainArea::set_weight(int p_index, const Matrix &p_matrix) {
	ERR_FAIL_INDEX(p_index, weights.size());
	weights[p_index] = p_matrix;
}

void brain::BrainArea::set_biases(int p_index, const Matrix &p_matrix) {
	ERR_FAIL_INDEX(p_index, biases.size());
	biases[p_index] = p_matrix;
}

void brain::BrainArea::set_activations(int p_index, Activation p_activation) {
	ERR_FAIL_INDEX(p_index, activations.size());
	activations[p_index] = p_activation;
}

real_t brain::BrainArea::learn(
		const Matrix &p_input,
		const Matrix &p_expected,
		real_t p_learn_rate,
		LearningCache *p_cache) {

	ERR_FAIL_COND_V(p_input.get_rows() != get_layer_size(INPUT_LAYER_ID), 10000);
	ERR_FAIL_COND_V(p_input.get_columns() != 1, 10000);
	ERR_FAIL_COND_V(p_expected.get_rows() != get_layer_size(OUTPUT_LAYER_ID), 10000);
	ERR_FAIL_COND_V(p_expected.get_columns() != 1, 10000);

	const bool shared_cache = p_cache;
	if (!shared_cache) {
		p_cache = new LearningCache;
	}

	brain::Matrix guess_res;
	guess(p_input, guess_res, p_cache);

	const Matrix total_error = p_expected - guess_res;
	Matrix propagated_error = total_error;

	Matrix layer_error;

	/// Back propagation

	/// The first task of this cycle is to calculate the error for the
	/// layer - 1  before updates the weights.
	/// Then calculate the gradient error and apply this gradient to the weight
	for (int l(get_layer_count() - 1); 1 <= l; --l) {

		layer_error = propagated_error;

		// Propagate the error before change the weights
		// Skip propagation if we are on penultimate layer
		if (1 <= l - 1)
			propagated_error =
					weights[WEIGHT_ID(l - 1)].transposed() * propagated_error;

		// Calculate gradient
		Matrix &gradient = p_cache->layers_output[l];

		DEBUG_ONLY(ERR_FAIL_COND_V(activations[ACTIVATION_ID(l)] == ACTIVATION_MAX, 10000));
		gradient.map(activation_derivatives[activations[ACTIVATION_ID(l)]]);

		gradient.element_wise_multiplicate(layer_error);
		gradient *= p_learn_rate;

		// Prepare the input for delta weight calc
		const Matrix input_transposed(p_cache->layers_output[l - 1].transposed());

		// Adjust weights by its delta
		weights[WEIGHT_ID(l - 1)] += (gradient * input_transposed);
		biases[BIAS_ID(l - 1)] += gradient;
	}

	if (!shared_cache) {
		delete p_cache;
		p_cache = nullptr;
	}

	// Total error = Σ((expected - guess)^2)
	return total_error.mapped(brain::Math::pow, 2).summation();
}

void brain::BrainArea::guess(
		const Matrix &p_input,
		Matrix &r_guess,
		LearningCache *p_cache) {

	ERR_FAIL_COND(p_input.get_rows() != get_layer_size(INPUT_LAYER_ID));
	ERR_FAIL_COND(p_input.get_columns() != 1);

	if (p_cache)
		p_cache->layers_output.resize(get_layer_count());

	r_guess = p_input;

	for (int i(0); i < weights.size(); ++i) {

		if (p_cache)
			p_cache->layers_output[i] = r_guess;

		// Layer calculation
		r_guess = (weights[i] * r_guess) + biases[i];

		// Activation of next layer
		DEBUG_ONLY(ERR_FAIL_COND(activations[ACTIVATION_ID(i + 1)] == ACTIVATION_MAX));
		r_guess.map(activation_functions[activations[ACTIVATION_ID(i + 1)]]);
	}

	if (p_cache)
		p_cache->layers_output[get_layer_count() - 1] = r_guess;
}

void brain::BrainArea::set_layer_size(uint32_t p_layer, uint32_t p_size) {
	ERR_FAIL_INDEX(p_layer, OUTPUT_LAYER_ID + 1);

	// Update previous weight layer size
	if (0 < p_layer) {
		weights[p_layer - 1].resize(
				p_size,
				get_layer_size(p_layer - 1));

		biases[p_layer - 1].resize(p_size, 1);
	}

	// Update this weight layer size
	if (p_layer < weights.size()) {

		weights[p_layer].resize(
				get_layer_size(p_layer + 1),
				p_size);
	}
}

uint32_t brain::BrainArea::get_layer_size(uint32_t p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, weights.size() + 1, 0);

	if (p_layer == OUTPUT_LAYER_ID) {
		// This happens only for the last layer
		return weights[p_layer - 1].get_rows();
	} else {
		return weights[p_layer].get_columns();
	}
}
