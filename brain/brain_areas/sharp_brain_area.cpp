#include "sharp_brain_area.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

//typedef real_t (*activation_func)(real_t p_val);
//
//activation_func activation_functions[] = { brain::Math::sigmoid };
//activation_func activation_derivatives[] = { brain::Math::sigmoid_fast_derivative };

void brain::Neuron::force_set_value(real_t p_val, uint32_t p_execution_id) const {
	execution_id = p_execution_id;
	cached_value = p_val;
}

real_t brain::Neuron::get_value(uint32_t p_execution_id) const {
	if (execution_id != p_execution_id) {
		execution_id = p_execution_id;
		cached_value = 0.f;
		for (auto it = parents.begin(); it != parents.end(); ++it) {
			cached_value += it->neuron->get_value(p_execution_id) * it->weight;
		}
		cached_value = brain::BrainArea::activation_functions[activation](cached_value);
	}
	return cached_value;
}

brain::Neuron::Neuron(uint32_t p_id) :
		id(p_id),
		execution_id(0),
		activation(BrainArea::ACTIVATION_SIGMOID) {
}

bool brain::Neuron::has_parent(Neuron *p_neuron) const {
	for (auto it = parents.begin(); it != parents.end(); ++it) {
		if (it->neuron == p_neuron)
			return true;
	}
	return false;
}

uint32_t brain::Neuron::get_parent_count() const {
	return parents.size();
}

void brain::Neuron::add_parent(Neuron *p_neuron, real_t p_weight) {

	ERR_FAIL_COND(has_parent(p_neuron));
	ERR_FAIL_COND(p_neuron->has_parent(this));

	parents.push_back({ p_neuron, p_weight });
}

void brain::Neuron::set_weight(uint32_t p_parent_index, real_t p_weight) {
	ERR_FAIL_INDEX(p_parent_index, parents.size());

	parents[p_parent_index].weight = p_weight;
}

brain::SharpBrainArea::SharpBrainArea() :
		brain::BrainArea(brain::BRAIN_AREA_TYPE_SHARP),
		execution_id(0),
		ready(false) {}

uint32_t brain::SharpBrainArea::add_neuron() {
	const uint32_t id(neurons.size());
	neurons.push_back(Neuron(id));
	return id;
}

bool brain::SharpBrainArea::is_neuron_input(uint32_t p_neuron_id) const {
	ERR_FAIL_INDEX_V(p_neuron_id, neurons.size(), false);
	for (auto it = inputs.begin(); it != inputs.end(); ++it) {
		if (*it == p_neuron_id)
			return true;
	}
	return false;
}

void brain::SharpBrainArea::set_neuron_as_input(uint32_t p_neuron_id) {
	ERR_FAIL_INDEX(p_neuron_id, neurons.size());
	ERR_FAIL_COND(is_neuron_input(p_neuron_id));
	ERR_FAIL_COND(is_neuron_output(p_neuron_id));
	inputs.push_back(p_neuron_id);
	ready = false;
}

bool brain::SharpBrainArea::is_neuron_output(uint32_t p_neuron_id) const {
	ERR_FAIL_INDEX_V(p_neuron_id, neurons.size(), false);
	for (auto it = outputs.begin(); it != outputs.end(); ++it) {
		if (*it == p_neuron_id)
			return true;
	}
	return false;
}

void brain::SharpBrainArea::set_neuron_as_output(uint32_t p_neuron_id) {
	ERR_FAIL_INDEX(p_neuron_id, neurons.size());
	ERR_FAIL_COND(is_neuron_input(p_neuron_id));
	ERR_FAIL_COND(is_neuron_output(p_neuron_id));
	outputs.push_back(p_neuron_id);
	ready = false;
}

void brain::SharpBrainArea::add_link(
		uint32_t p_neuron_parent_id,
		uint32_t p_neuron_child_id,
		real_t p_weight) {

	ERR_FAIL_INDEX(p_neuron_parent_id, neurons.size());
	ERR_FAIL_INDEX(p_neuron_child_id, neurons.size());

	neurons[p_neuron_child_id].add_parent(
			&neurons[p_neuron_parent_id],
			p_weight);
	ready = false;
}

void brain::SharpBrainArea::clear() {
	inputs.clear();
	outputs.clear();
	neurons.clear();
	ready = false;
}

void brain::SharpBrainArea::randomize_weights(real_t p_range) {
	// If not ready check it
	if (!ready) {
		SharpBrainArea *mutable_this = const_cast<SharpBrainArea *>(this);
		mutable_this->check_if_ready();
		ERR_FAIL_COND(!ready);
	}

	for (int i(outputs.size() - 1); 0 <= i; ++i) {

		randomize_parents_weight(&neurons[outputs[i]], p_range);
	}
}

void brain::SharpBrainArea::fill_weights(real_t p_weight) {

	// If not ready check it
	if (!ready) {
		SharpBrainArea *mutable_this = const_cast<SharpBrainArea *>(this);
		mutable_this->check_if_ready();
		ERR_FAIL_COND(!ready);
	}

	for (int i(outputs.size() - 1); 0 <= i; ++i) {

		set_parents_weight(&neurons[outputs[i]], p_weight);
	}
}

uint32_t brain::SharpBrainArea::get_input_layer_size() const {
	return inputs.size();
}

uint32_t brain::SharpBrainArea::get_output_layer_size() const {
	return outputs.size();
}

void brain::SharpBrainArea::guess(
		const Matrix &p_input,
		Matrix &r_guess) const {

	r_guess.resize(outputs.size(), 1);

	// If not ready check it
	if (!ready) {
		SharpBrainArea *mutable_this = const_cast<SharpBrainArea *>(this);
		mutable_this->check_if_ready();
		ERR_FAIL_COND(!ready);
	}

	ERR_FAIL_COND(p_input.get_row_count() != inputs.size());
	ERR_FAIL_COND(p_input.get_column_count() != 1);

	++execution_id;

	// set inputs
	for (int i(inputs.size() - 1); 0 <= i; --i) {
		neurons[inputs[i]].force_set_value(p_input.get(i, 0), execution_id);
	}

	// Get outputs
	for (int i(outputs.size() - 1); 0 <= i; --i) {
		const real_t val = neurons[outputs[i]].get_value(execution_id);
		r_guess.set(i, 0, val);
	}
}

bool brain::SharpBrainArea::is_fully_linked_to_input(Neuron *p_neuron) const {

	// This happens only to the input layer
	if (is_neuron_input(p_neuron->id))
		return true;

	if (!p_neuron->parents.size())
		return false;

	// This check if this neuron in the hidden layer is fully connected to input
	for (auto p_it = p_neuron->parents.begin();
			p_it != p_neuron->parents.end();
			++p_it) {

		if (!is_fully_linked_to_input(p_it->neuron)) {
			ERR_EXPLAIN("The neuron is not fully connected to the input. Neuron ID: " + brain::itos(p_it->neuron->id));
			ERR_FAIL_V(false);
		}
	}
	return true;
}

void brain::SharpBrainArea::check_if_ready() {
	ready = false;

	ERR_FAIL_COND(!get_input_layer_size());
	ERR_FAIL_COND(!get_output_layer_size());

	// Check if the output neurons are fully connected to the inputs
	for (auto o_it = outputs.begin(); o_it != outputs.end(); ++o_it) {
		if (!is_fully_linked_to_input(&neurons[*o_it]))
			return;
	}

	ready = true;
}

void brain::SharpBrainArea::randomize_parents_weight(
		Neuron *p_neuron,
		real_t p_range) {

	for (auto p_it = p_neuron->parents.begin();
			p_it != p_neuron->parents.end();
			++p_it) {

		p_it->weight = brain::Math::random(-p_range, p_range);
		randomize_parents_weight(p_it->neuron, p_range);
	}
}

void brain::SharpBrainArea::set_parents_weight(
		Neuron *p_neuron,
		real_t p_weight) {

	for (auto p_it = p_neuron->parents.begin();
			p_it != p_neuron->parents.end();
			++p_it) {

		p_it->weight = p_weight;
		set_parents_weight(p_it->neuron, p_weight);
	}
}
