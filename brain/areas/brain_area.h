#pragma once

#include "brain/math/dynamic_matrix.h"
#include <vector>

namespace brain {
class BrainArea {

	DynamicMatrix input_layer;
	DynamicMatrix input_weights;
	DynamicMatrix output_layer;
	DynamicMatrix output_weights;
	std::vector<DynamicMatrix> hidden_layers;
	bool ready;

public:
	BrainArea();

	void set_input_layer_size(int p_size);
	int get_input_layer_size() const;

	void set_output_layer_size(int p_size);
	int get_output_layer_size() const;

	void set_hidden_layers_count(int p_count);

	void set_hidden_layer_size(int p_layer, int p_size);
	int get_hidden_layer_size(int p_layer) const;

	bool prepare();
};
} // namespace brain
