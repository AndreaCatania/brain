#include "uniform_brain_area.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

#define HIDDEN_LAYER(layer) (layer + 1)
#define HIDDEN_LAYER_COUNT(count) (count - 2)

#define INPUT_LAYER_ID 0
#define OUTPUT_LAYER_ID weights.size()

#define ACTIVATION_ID(layer) ((layer)-1)
#define WEIGHT_ID(layer) (layer)
#define BIAS_ID(layer) (layer)

brain::UniformBrainArea::UniformBrainArea() :
		brain::BrainArea(BRAIN_AREA_TYPE_UNIFORM) {
	weights.resize(1);
	biases.resize(1);
	activations.push_back(ACTIVATION_SIGMOID);
}

brain::UniformBrainArea::UniformBrainArea(
		uint32_t p_input_layer_size,
		uint32_t p_hidden_layers_count,
		uint32_t p_output_layer_size) :
		UniformBrainArea() {
	set_input_layer_size(p_input_layer_size);
	set_hidden_layers_count(p_hidden_layers_count);
	set_output_layer_size(p_output_layer_size);
}

void brain::UniformBrainArea::set_input_layer_size(uint32_t p_size) {
	set_layer_size(INPUT_LAYER_ID, p_size);
}

uint32_t brain::UniformBrainArea::get_input_layer_size() const {
	return get_layer_size(INPUT_LAYER_ID);
}

void brain::UniformBrainArea::set_output_layer_size(uint32_t p_size) {

	set_layer_size(OUTPUT_LAYER_ID, p_size);
}

uint32_t brain::UniformBrainArea::get_output_layer_size() const {
	return get_layer_size(OUTPUT_LAYER_ID);
}

void brain::UniformBrainArea::set_hidden_layers_count(uint32_t p_count) {

	const int prev_size_output_layer = get_layer_size(OUTPUT_LAYER_ID);
	const Activation prev_activ_output_layer = activations[activations.size() - 1];

	weights.resize(p_count + 2 - 1);
	biases.resize(p_count + 2 - 1);
	activations.resize(p_count + 2 - 1);

	set_layer_size(OUTPUT_LAYER_ID, prev_size_output_layer);
	activations[activations.size() - 1] = prev_activ_output_layer;
}

uint32_t brain::UniformBrainArea::get_hidden_layers_count() const {
	return weights.size() - 1;
}

void brain::UniformBrainArea::set_hidden_layer(
		uint32_t p_hidden_layer,
		uint32_t p_size,
		Activation p_activation) {

	set_hidden_layer_size(p_hidden_layer, p_size);
	set_hidden_layer_activation(p_hidden_layer, p_activation);
}

void brain::UniformBrainArea::set_hidden_layer_size(uint32_t p_hidden_layer, uint32_t p_size) {
	set_layer_size(HIDDEN_LAYER(p_hidden_layer), p_size);
}

uint32_t brain::UniformBrainArea::get_hidden_layer_size(uint32_t p_hidden_layer) const {
	return get_layer_size(HIDDEN_LAYER(p_hidden_layer));
}

void brain::UniformBrainArea::set_hidden_layer_activation(uint32_t p_hidden_layer, Activation p_activation) {
	ERR_FAIL_INDEX(p_hidden_layer, activations.size());
	activations[p_hidden_layer] = p_activation;
}

brain::UniformBrainArea::Activation brain::UniformBrainArea::get_hidden_layer_activation(
		uint32_t p_hidden_layer) const {
	ERR_FAIL_INDEX_V(p_hidden_layer, activations.size(), ACTIVATION_MAX);
	return activations[p_hidden_layer];
}

real_t matrix_rand(real_t x, real_t p_range) {
	return brain::Math::random(-p_range, p_range);
}

void brain::UniformBrainArea::randomize_weights(real_t p_range) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].map(matrix_rand, p_range);
	}
}

void brain::UniformBrainArea::fill_weights(real_t p_value) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].set_all(p_value);
	}
}

void brain::UniformBrainArea::randomize_biases(real_t p_range) {

	for (int i(0); i < biases.size(); ++i) {
		biases[i].map(matrix_rand, p_range);
	}
}

void brain::UniformBrainArea::fill_biases(real_t p_value) {
	for (int i(0); i < biases.size(); ++i) {
		biases[i].set_all(p_value);
	}
}

int brain::UniformBrainArea::get_layer_count() const {
	return weights.size() + 1;
}

const brain::Matrix &brain::UniformBrainArea::get_layer_weights(const int p_layer) const {
	return weights[p_layer];
}

void brain::UniformBrainArea::set_weight(int p_index, const Matrix &p_matrix) {
	ERR_FAIL_INDEX(p_index, weights.size());
	weights[p_index] = p_matrix;
}

void brain::UniformBrainArea::set_biases(int p_index, const Matrix &p_matrix) {
	ERR_FAIL_INDEX(p_index, biases.size());
	biases[p_index] = p_matrix;
}

void brain::UniformBrainArea::set_activations(int p_index, Activation p_activation) {
	ERR_FAIL_INDEX(p_index, activations.size());
	activations[p_index] = p_activation;
}

real_t brain::UniformBrainArea::learn(
		const Matrix &p_input,
		const Matrix &p_expected,
		real_t p_learn_rate,
		LearningCache *p_cache) {

	ERR_FAIL_COND_V(p_input.get_row_count() != get_layer_size(INPUT_LAYER_ID), 10000);
	ERR_FAIL_COND_V(p_input.get_column_count() != 1, 10000);
	ERR_FAIL_COND_V(p_expected.get_row_count() != get_layer_size(OUTPUT_LAYER_ID), 10000);
	ERR_FAIL_COND_V(p_expected.get_column_count() != 1, 10000);

	const bool shared_cache = p_cache;
	if (!shared_cache) {
		p_cache = new LearningCache;
	}

	brain::Matrix guess_res;
	_guess(p_input, guess_res, p_cache);

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

bool brain::UniformBrainArea::guess(
		const Matrix &p_input,
		Matrix &r_guess) const {

	return _guess(p_input, r_guess);
}

bool brain::UniformBrainArea::_guess(
		const Matrix &p_input,
		Matrix &r_guess,
		LearningCache *p_cache) const {

	ERR_FAIL_COND_V(p_input.get_row_count() != get_layer_size(INPUT_LAYER_ID), false);
	ERR_FAIL_COND_V(p_input.get_column_count() != 1, false);

	if (p_cache)
		p_cache->layers_output.resize(get_layer_count());

	r_guess = p_input;

	for (int i(0); i < weights.size(); ++i) {

		if (p_cache)
			p_cache->layers_output[i] = r_guess;

		// Layer calculation
		r_guess = (weights[i] * r_guess) + biases[i];

		// Activation of next layer
		DEBUG_ONLY(ERR_FAIL_COND_V(activations[ACTIVATION_ID(i + 1)] == ACTIVATION_MAX, false));
		r_guess.map(activation_functions[activations[ACTIVATION_ID(i + 1)]]);
	}

	if (p_cache)
		p_cache->layers_output[get_layer_count() - 1] = r_guess;

	return true;
}

int brain::UniformBrainArea::get_buffer_metadata_size() const {
	return sizeof(uint32_t) * METADATA_MAX; // Metadata size
}

uint32_t brain::UniformBrainArea::get_buffer_size(const std::vector<uint8_t> &p_buffer_metadata) const {
	const uint32_t buffer_size = ((uint32_t *)p_buffer_metadata.data())[METADATA_BUFFER_SIZE];
	return buffer_size;
}

bool brain::UniformBrainArea::is_buffer_corrupted(const std::vector<uint8_t> &p_buffer) const {

	const uint32_t buffer_size = ((uint32_t *)p_buffer.data())[METADATA_BUFFER_SIZE];
	const uint32_t real_size = ((uint32_t *)p_buffer.data())[METADATA_REAL_SIZE];
	const uint32_t weight_count = ((uint32_t *)p_buffer.data())[METADATA_WEIGHT_COUNT];
	const uint32_t bias_count = ((uint32_t *)p_buffer.data())[METADATA_BIAS_COUNT];
	const uint32_t activation_count = ((uint32_t *)p_buffer.data())[METADATA_ACTIVATION_COUNT];

	ERR_FAIL_COND_V(p_buffer.size() != buffer_size, true);
	ERR_FAIL_COND_V(sizeof(float) != real_size && sizeof(double) != real_size, true);
	ERR_FAIL_COND_V(weight_count != bias_count, true);
	ERR_FAIL_COND_V(weight_count != activation_count, true);

	return false;
}

bool brain::UniformBrainArea::is_buffer_compatible(const std::vector<uint8_t> &p_buffer) const {

	const uint32_t real_size = ((uint32_t *)p_buffer.data())[METADATA_REAL_SIZE];
	const uint32_t weight_count = ((uint32_t *)p_buffer.data())[METADATA_WEIGHT_COUNT];
	const uint32_t bias_count = ((uint32_t *)p_buffer.data())[METADATA_BIAS_COUNT];
	const uint32_t activation_count = ((uint32_t *)p_buffer.data())[METADATA_ACTIVATION_COUNT];

	ERR_FAIL_COND_V(is_buffer_corrupted(p_buffer), false);

	if (
			weights.size() != weight_count ||
			biases.size() != bias_count ||
			activations.size() != activation_count)
		return false;

	const uint8_t *b_support = p_buffer.data() + get_buffer_metadata_size();

	Matrix m;
	for (int i(0); i < weights.size(); ++i) {
		m.from_byte(b_support, real_size);
		const size_t matrix_size = weights[i].get_byte_size();
		b_support += matrix_size;

		if (
				m.get_row_count() != weights[i].get_row_count() ||
				m.get_column_count() != weights[i].get_column_count())
			return false;
	}

	return true;
}

bool brain::UniformBrainArea::set_buffer(const std::vector<uint8_t> &p_buffer) {

	// Read metadata
	const uint32_t real_size = ((uint32_t *)p_buffer.data())[METADATA_REAL_SIZE];
	const uint32_t weight_count = ((uint32_t *)p_buffer.data())[METADATA_WEIGHT_COUNT];
	const uint32_t bias_count = ((uint32_t *)p_buffer.data())[METADATA_BIAS_COUNT];
	const uint32_t activation_count = ((uint32_t *)p_buffer.data())[METADATA_ACTIVATION_COUNT];

	ERR_FAIL_COND_V(is_buffer_corrupted(p_buffer), false);

	weights.resize(weight_count);
	biases.resize(bias_count);
	activations.resize(activation_count);

	const uint8_t *b_support = p_buffer.data() + get_buffer_metadata_size();

	for (int i(0); i < weights.size(); ++i) {
		weights[i].from_byte(b_support, real_size);
		const size_t matrix_size = weights[i].get_byte_size();
		b_support += matrix_size;
	}

	for (int i(0); i < biases.size(); ++i) {
		biases[i].from_byte(b_support, real_size);
		const size_t matrix_size = biases[i].get_byte_size();
		b_support += matrix_size;
	}

	for (int i(0); i < activations.size(); ++i) {
		activations[i] = *((const Activation *)b_support);
		b_support += sizeof(int);
	}

	return true;
}

bool brain::UniformBrainArea::get_buffer(std::vector<uint8_t> &r_buffer) const {

	const int real_size = sizeof(real_t);

	uint32_t buffer_size = get_buffer_metadata_size();

	for (int i(0); i < weights.size(); ++i) {
		buffer_size += weights[i].get_byte_size();
	}
	for (int i(0); i < biases.size(); ++i) {
		buffer_size += biases[i].get_byte_size();
	}
	buffer_size += activations.size() * sizeof(int);

	r_buffer.resize(buffer_size);

	((uint32_t *)r_buffer.data())[METADATA_BUFFER_SIZE] = buffer_size;
	((uint32_t *)r_buffer.data())[METADATA_REAL_SIZE] = real_size;
	((uint32_t *)r_buffer.data())[METADATA_WEIGHT_COUNT] = weights.size();
	((uint32_t *)r_buffer.data())[METADATA_BIAS_COUNT] = biases.size();
	((uint32_t *)r_buffer.data())[METADATA_ACTIVATION_COUNT] = activations.size();

	uint8_t *b_support = r_buffer.data() + get_buffer_metadata_size();

	for (int i(0); i < weights.size(); ++i) {
		weights[i].to_byte(b_support);
		const size_t matrix_size = weights[i].get_byte_size();
		b_support += matrix_size;
	}

	for (int i(0); i < biases.size(); ++i) {
		biases[i].to_byte(b_support);
		const size_t matrix_size = biases[i].get_byte_size();
		b_support += matrix_size;
	}

	for (int i(0); i < activations.size(); ++i) {
		*((int *)b_support) = activations[i];
		b_support += sizeof(int);
	}

	return true;
}

void brain::UniformBrainArea::set_layer_size(uint32_t p_layer, uint32_t p_size) {
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

uint32_t brain::UniformBrainArea::get_layer_size(uint32_t p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, weights.size() + 1, 0);

	if (p_layer == OUTPUT_LAYER_ID) {
		// This happens only for the last layer
		return weights[p_layer - 1].get_row_count();
	} else {
		return weights[p_layer].get_column_count();
	}
}
