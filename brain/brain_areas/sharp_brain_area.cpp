#include "sharp_brain_area.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include <algorithm>

void brain::Neuron::force_set_value(real_t p_val, uint32_t p_execution_id) const {
	execution_id = p_execution_id;
	cached_value = p_val;
}

real_t brain::Neuron::get_value(uint32_t p_execution_id) const {
	if (execution_id != p_execution_id) {
		real_t value = 0.f;
		for (auto it = parents.begin(); it != parents.end(); ++it) {
			if (it->is_recurrent)
				value += it->neuron->get_recurrent(p_execution_id) * it->weight;
			else
				value += it->neuron->get_value(p_execution_id) * it->weight;
		}

		recurrent = cached_value;

		if (activation == brain::BrainArea::ACTIVATION_SOFTMAX)
			// Softmax activation is performed out of this function and works only
			// for the output neurons
			cached_value = value;
		else
			cached_value = brain::BrainArea::activation_functions[activation](value);

		execution_id = p_execution_id;
	}
	return cached_value;
}

real_t brain::Neuron::get_recurrent(uint32_t p_execution_id) const {
	return execution_id == p_execution_id ? recurrent : cached_value;
}

brain::Neuron::Neuron(NeuronId p_id) :
		id(p_id),
		cached_value(0),
		recurrent(0),
		execution_id(0),
		activation(BrainArea::ACTIVATION_SIGMOID) {
}

void brain::Neuron::prepare_internal_memory(SharpBrainArea *p_owner) {
	for (auto it = parents.begin(); it != parents.end(); ++it) {
		it->neuron = &*(p_owner->neurons.begin() + it->neuron_id);
	}
}

bool brain::Neuron::has_parent(NeuronId p_neuron_id) const {
	for (auto it = parents.begin(); it != parents.end(); ++it) {
		if (it->neuron_id == p_neuron_id)
			return true;
	}
	return false;
}

uint32_t brain::Neuron::get_parent_count() const {
	return parents.size();
}

void brain::Neuron::add_parent(NeuronId p_neuron_id, real_t p_weight, bool p_recurrent) {

	parents.push_back({ nullptr, p_neuron_id, p_weight, p_recurrent });
}

void brain::Neuron::set_weight(uint32_t p_parent_index, real_t p_weight) {
	ERR_FAIL_INDEX(p_parent_index, parents.size());

	parents[p_parent_index].weight = p_weight;
}

#include "brain/NEAT/neat_organism.h"

brain::SharpBrainArea::SharpBrainArea(NtOrganism *p) :
		o(p),
		brain::BrainArea(brain::BRAIN_AREA_TYPE_SHARP),
		execution_id(0),
		ready(false) {}

brain::NeuronId brain::SharpBrainArea::add_neuron() {
	const NeuronId id(neurons.size());
	neurons.push_back(Neuron(id));
	ready = false;
	return id;
}

int brain::SharpBrainArea::get_neuron_count() const {
	return neurons.size();
}

bool brain::SharpBrainArea::is_neuron_input(NeuronId p_neuron_id) const {
	ERR_FAIL_INDEX_V(p_neuron_id, neurons.size(), false);
	for (auto it = inputs.begin(); it != inputs.end(); ++it) {
		if (*it == p_neuron_id)
			return true;
	}
	return false;
}

void brain::SharpBrainArea::set_neuron_as_input(NeuronId p_neuron_id) {
	ERR_FAIL_INDEX(p_neuron_id, neurons.size());
	ERR_FAIL_COND(is_neuron_input(p_neuron_id));
	ERR_FAIL_COND(is_neuron_output(p_neuron_id));
	inputs.push_back(p_neuron_id);
	ready = false;
}

bool brain::SharpBrainArea::is_neuron_output(NeuronId p_neuron_id) const {
	ERR_FAIL_INDEX_V(p_neuron_id, neurons.size(), false);
	for (auto it = outputs.begin(); it != outputs.end(); ++it) {
		if (*it == p_neuron_id)
			return true;
	}
	return false;
}

void brain::SharpBrainArea::set_neuron_as_output(NeuronId p_neuron_id) {
	ERR_FAIL_INDEX(p_neuron_id, neurons.size());
	ERR_FAIL_COND(is_neuron_input(p_neuron_id));
	ERR_FAIL_COND(is_neuron_output(p_neuron_id));
	outputs.push_back(p_neuron_id);
	ready = false;
}

void brain::SharpBrainArea::set_neuron_activation(
		NeuronId p_neuron_id,
		Activation p_activation) {

	ERR_FAIL_INDEX(p_neuron_id, neurons.size());
	neurons[p_neuron_id].activation = p_activation;
}

brain::BrainArea::Activation brain::SharpBrainArea::get_neuron_activation(
		NeuronId p_neuron_id) const {

	ERR_FAIL_INDEX_V(p_neuron_id, neurons.size(), ACTIVATION_MAX);
	return neurons[p_neuron_id].activation;
}

void brain::SharpBrainArea::add_link(
		NeuronId p_neuron_parent_id,
		NeuronId p_neuron_child_id,
		real_t p_weight,
		bool p_recurrent) {

	ERR_FAIL_INDEX(p_neuron_parent_id, neurons.size());
	ERR_FAIL_INDEX(p_neuron_child_id, neurons.size());

	Neuron *child = &*(neurons.begin() + p_neuron_child_id);

	ERR_FAIL_COND(child->has_parent(p_neuron_parent_id));
	ERR_FAIL_COND(!p_recurrent && p_neuron_parent_id == p_neuron_child_id);

	child->add_parent(
			p_neuron_parent_id,
			p_weight,
			p_recurrent);

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
		mutable_this->check_ready();
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
		mutable_this->check_ready();
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

bool brain::SharpBrainArea::guess(
		const Matrix &p_input,
		Matrix &r_guess) const {

	const int output_size = outputs.size();
	ERR_FAIL_COND_V(!output_size, false);

	r_guess.resize(output_size, 1);

	// If not ready check it
	if (!ready) {
		SharpBrainArea *mutable_this = const_cast<SharpBrainArea *>(this);
		mutable_this->check_ready();
		ERR_FAIL_COND_V(!ready, false);
	}

	ERR_FAIL_COND_V(p_input.get_row_count() != inputs.size(), false);
	ERR_FAIL_COND_V(p_input.get_column_count() != 1, false);

	++execution_id;

	// set inputs
	for (int i(inputs.size() - 1); 0 <= i; --i) {
		neurons[inputs[i]].force_set_value(p_input.get(i, 0), execution_id);
	}

	// Get outputs
	for (int i(0); i < output_size; ++i) {
		const real_t val = neurons[outputs[i]].get_value(execution_id);
		r_guess.set(i, 0, val);
	}

	// Special case for softmax activation function
	if (ACTIVATION_SOFTMAX == neurons[outputs[0]].activation) {

		const real_t sum_exp(brain::Math::pow(Math_E, r_guess.summation()));
		for (int i(0); i < output_size; ++i) {

			const real_t v = brain::Math::soft_max_fast(
					neurons[outputs[i]].get_value(execution_id),
					sum_exp);
			neurons[outputs[i]].force_set_value(v, execution_id);
			r_guess.set(i, 0, v);
		}
	}

	return true;
}

bool brain::SharpBrainArea::are_links_walkable(
		Neuron *p_neuron,
		bool p_error_on_broken_link,
		bool p_error_on_dead_branches,
		std::vector<NeuronId> &r_cache) const {

	// This happens only to the input layer
	if (is_neuron_input(p_neuron->id))
		return true;

	// Check dead branches
	if (p_error_on_dead_branches) {
		ERR_FAIL_COND_V(!p_neuron->parents.size(), false);
	} else if (!p_neuron->parents.size())
		return true;

	r_cache.push_back(p_neuron->id);

	bool failed = false;
	std::string explain;

	// This check if this neuron in the hidden layer is fully connected to input
	for (auto p_it = p_neuron->parents.begin();
			p_it != p_neuron->parents.end();
			++p_it) {

		if (p_it->is_recurrent)
			continue;

		if (r_cache.end() != std::find(r_cache.begin(), r_cache.end(), p_it->neuron_id)) {
			explain = "Just detected a loop in the network. between these " + itos(reinterpret_cast<uint64_t>(this)) + " neurons:";
			for (auto it = r_cache.begin(); it != r_cache.end(); ++it) {
				explain += "\n" + itos(*it);
			}

			failed = true;
			break;
		}

		if (!are_links_walkable(
					p_it->neuron,
					p_error_on_broken_link,
					p_error_on_dead_branches,
					r_cache)) {

			if (p_error_on_broken_link) {
				explain = "The neuron is not fully connected to the input. Neuron ID: " + brain::itos(p_it->neuron->id);
				failed = true;
				break;
			} else {
				// Already loggeed, just report it up
				failed = true;
				break;
			}
		}
	}

	auto it = std::find(r_cache.begin(), r_cache.end(), p_neuron->id);
	ERR_FAIL_COND_V(it == r_cache.end(), false);
	r_cache.erase(it);

	if (failed) {
		if (explain == "") {
			// Alreayd logged
			return false;
		}
		ERR_EXPLAIN(explain);
		ERR_FAIL_V(false);
	}

	return true;
}

void brain::SharpBrainArea::check_ready() {
	ready = false;

	ERR_FAIL_COND(!get_input_layer_size());
	ERR_FAIL_COND(!get_output_layer_size());

	for (auto it = neurons.begin(); it != neurons.end(); ++it) {
		it->prepare_internal_memory(this);
	}

	std::vector<NeuronId> cache;

	// Check if the output neurons are fully connected to the inputs
	for (auto o_it = outputs.begin(); o_it != outputs.end(); ++o_it) {
		if (!are_links_walkable(&neurons[*o_it], false, false, cache))
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
