#include "brain_area.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

#define HIDDEN_LAYER(layer) (layer + 1)
#define HIDDEN_LAYER_COUNT(count) (count - 2)
#define INPUT_LAYER_INDEX 0
#define OUTPUT_LAYER_INDEX weights.size()
#define ERROR_ID(layer) ((layer)-1)

matrix_map activation_functions[] = { brain::Math::sigmoid };
matrix_map derivatives_functions[] = { brain::Math::sigmoid };

brain::BrainArea::BrainArea() {
	weights.resize(1);
	biases.resize(1);
	activations.push_back(ACTIVATION_SIGMOID);
}

void brain::BrainArea::set_input_layer_size(uint32_t p_size) {
	set_layer_size(INPUT_LAYER_INDEX, p_size);
}

uint32_t brain::BrainArea::get_input_layer_size() const {
	return get_layer_size(INPUT_LAYER_INDEX);
}

void brain::BrainArea::set_output_layer_size(uint32_t p_size) {

	set_layer_size(OUTPUT_LAYER_INDEX, p_size);
}

uint32_t brain::BrainArea::get_output_layer_size() const {
	return get_layer_size(OUTPUT_LAYER_INDEX);
}

void brain::BrainArea::resize_hidden_layers(uint32_t p_count) {

	const int prev_size_output_layer = get_layer_size(OUTPUT_LAYER_INDEX);
	const Activation prev_activ_output_layer = activations[activations.size() - 1];

	weights.resize(p_count + 2 - 1);
	biases.resize(p_count + 2 - 1);
	activations.resize(p_count + 2 - 1);

	set_layer_size(OUTPUT_LAYER_INDEX, prev_size_output_layer);
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

void brain::BrainArea::set_weights(real_t p_value) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].set_all(p_value);
	}
}

void brain::BrainArea::randomize_biases(real_t p_range) {

	for (int i(0); i < biases.size(); ++i) {
		biases[i].map(matrix_rand, p_range);
	}
}

void brain::BrainArea::set_biases(real_t p_value) {
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

// TODO please remove this
void _print_line(const std::string &p_msg) {
	printf("[INFO] ");
	printf(p_msg.c_str());
	printf("\n");
}

real_t brain::BrainArea::learn(
		const Matrix &p_input,
		const Matrix &p_expected,
		real_t p_lear_rate) {

	// Initialize here to avoid initialize errors when not needed to learn
	errors.resize(get_layer_count() - 1);

	brain::Matrix guess_res;
	guess(p_input, guess_res);

	errors[errors.size() - 1] = p_expected - guess_res;

	// Back propagate error inside the hidden layers
	// The output layer has already the error so skip it
	// The first layer can't be wrong so skip it
	for (int l(get_layer_count() - 2); 1 <= l; --l) {

		errors[ERROR_ID(l)] = get_layer_weights(l).transposed() * errors[ERROR_ID(l + 1)];
	}

	// Adjust weights

	// TODO remove this test
	_print_line("Guess " + std::string(guess_res));
	_print_line("Expected " + std::string(p_expected));

	_print_line("Propagated errors ");
	for (int l(0); l < errors.size(); ++l) {
		_print_line("Hidden layer " + brain::itos(l) + " " + std::string(errors[l]));
	}

	// Total error = Σ((expected - guess)^2)
	return errors[errors.size() - 1].mapped(brain::Math::pow, 2).summation();
}

void brain::BrainArea::learn_cache_clear() {
	errors.resize(0);
}

void brain::BrainArea::guess(
		const Matrix &p_input,
		Matrix &r_guess) {

	ERR_FAIL_COND(p_input.get_rows() != get_layer_size(INPUT_LAYER_INDEX));
	ERR_FAIL_COND(p_input.get_columns() != 1);

	r_guess = p_input;

	for (int i(0); i < weights.size(); ++i) {

		// Layer calculation
		r_guess = weights[i] * r_guess + biases[i];

		// Activation of next layer
		DEBUG_ONLY(ERR_FAIL_COND(activations[i] == ACTIVATION_MAX));
		r_guess.map(activation_functions[activations[i]]);
	}
}

void brain::BrainArea::set_layer_size(uint32_t p_layer, uint32_t p_size) {
	ERR_FAIL_INDEX(p_layer, OUTPUT_LAYER_INDEX + 1);

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

	if (p_layer == OUTPUT_LAYER_INDEX) {
		// This happens only for the last layer
		return weights[p_layer - 1].get_rows();
	} else {
		return weights[p_layer].get_columns();
	}
}
