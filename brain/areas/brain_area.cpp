#include "brain_area.h"

#include "brain/error_macros.h"

brain::BrainArea::BrainArea() :
		ready(false) {}

void brain::BrainArea::set_input_layer_size(int p_size) {
	ready = false;
	input_layer.set_size(p_size, 1);
	input_weights.set_size();
}

int brain::BrainArea::get_input_layer_size() const {
	return input_layer.get_rows();
}

void brain::BrainArea::set_output_layer_size(int p_size) {
	ready = false;
	output_layer.set_size(p_size, 1);
}

int brain::BrainArea::get_output_layer_size() const {
	return output_layer.get_rows();
}

void brain::BrainArea::set_hidden_layers_count(int p_count) {
	ready = false;
	hidden_layers.resize(p_count);
}

void brain::BrainArea::set_hidden_layer_size(int p_layer, int p_size) {
	ERR_FAIL_INDEX(p_layer, hidden_layers.size());
	hidden_layers[p_layer].set_size(p_size, 1);
}

int brain::BrainArea::get_hidden_layer_size(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, hidden_layers.size(), 0);
	return hidden_layers[p_layer].get_rows();
}

bool brain::BrainArea::prepare() {

	return false;
}
