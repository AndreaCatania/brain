#pragma once

#include "brain/math/dynamic_matrix.h"
#include <vector>

namespace brain {
class BrainArea {

	std::vector<DynamicMatrix> weights;
	std::vector<DynamicMatrix> biases;

public:
	BrainArea();

	void set_input_layer_size(uint32_t p_size);
	uint32_t get_input_layer_size() const;

	void set_output_layer_size(uint32_t p_size);
	uint32_t get_output_layer_size() const;

	void resize_hidden_layers(uint32_t p_count);
	uint32_t get_hidden_layers_count() const;

	void set_hidden_layer_size(uint32_t p_layer, uint32_t p_size);
	uint32_t get_hidden_layer_size(uint32_t p_layer) const;

	void randomize_weights(real_t p_range, uint64_t *p_seed);
	void set_weights(real_t p_value);

	void set_biases(real_t p_value);

	void execute(const DynamicMatrix &p_input, DynamicMatrix &r_result);

private:
	void set_layer_size(uint32_t p_layer, uint32_t p_size);
	uint32_t get_layer_size(uint32_t p_layer) const;
};
} // namespace brain
