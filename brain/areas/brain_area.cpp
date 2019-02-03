#include "brain_area.h"

#include "brain/error_macros.h"

#define HIDDEN_LAYER(layer) (layer + 1)
#define HIDDEN_LAYER_COUNT(count) (count - 2)
#define INPUT_LAYER_INDEX 0
#define OUTPUT_LAYER_INDEX weights.size()

brain::BrainArea::BrainArea() {

	weights.resize(1);
	biases.resize(1);
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

	weights.resize(p_count + 2 - 1);
	biases.resize(p_count + 2 - 1);

	set_layer_size(OUTPUT_LAYER_INDEX, prev_size_output_layer);
}

uint32_t brain::BrainArea::get_hidden_layers_count() const {
	return weights.size() - 1;
}

void brain::BrainArea::set_hidden_layer_size(uint32_t p_layer, uint32_t p_size) {
	set_layer_size(HIDDEN_LAYER(p_layer), p_size);
}

uint32_t brain::BrainArea::get_hidden_layer_size(uint32_t p_layer) const {
	return get_layer_size(HIDDEN_LAYER(p_layer));
}

void brain::BrainArea::randomize_weights(real_t p_range, uint64_t *p_seed) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].randomize(p_range, p_seed);
	}
}

void brain::BrainArea::set_weights(real_t p_value) {

	for (int i(0); i < weights.size(); ++i) {
		weights[i].set_all(p_value);
	}
}

void brain::BrainArea::set_biases(real_t p_value) {
	for (int i(0); i < biases.size(); ++i) {
		biases[i].set_all(p_value);
	}
}

void brain::BrainArea::execute(
		const DynamicMatrix &p_input,
		DynamicMatrix &r_result) {

	ERR_FAIL_COND(p_input.get_rows() != get_layer_size(INPUT_LAYER_INDEX));
	ERR_FAIL_COND(p_input.get_columns() != 1);

	r_result = p_input;

	for (int i(0); i < weights.size(); ++i) {
		r_result = weights[i] * r_result + biases[i];
		r_result.sigmoid();
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
