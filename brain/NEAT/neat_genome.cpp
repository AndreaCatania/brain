#include "neat_genome.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include <algorithm>

brain::NtNeuronGene::NtNeuronGene(uint32_t p_id, NeuronGeneType p_type) :
		id(p_id),
		type(p_type) {}

brain::NtLinkGene::NtLinkGene() {}

brain::NtLinkGene::NtLinkGene(
		uint32_t p_id,
		bool p_active,
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id,
		real_t p_weight,
		bool p_recurrent,
		uint32_t p_innovation_number) :
		id(p_id),
		active(p_active),
		parent_neuron_id(p_parent_neuron_id),
		child_neuron_id(p_child_neuron_id),
		weight(p_weight),
		recurrent(p_recurrent),
		innovation_number(p_innovation_number) {}

brain::NtGenome::NtGenome() :
		biggest_innovation_number(0) {}

brain::NtGenome::NtGenome(
		int p_input_count,
		int p_output_count,
		bool p_randomize_weights) :
		NtGenome() {

	ERR_FAIL_COND(p_input_count <= 0);
	ERR_FAIL_COND(p_output_count <= 0);

	for (int i(0); i < p_input_count; ++i) {
		add_neuron(NtNeuronGene::NEURON_GENE_TYPE_INPUT);
	}

	for (int i(0); i < p_output_count; ++i) {
		add_neuron(NtNeuronGene::NEURON_GENE_TYPE_OUTPUT);
	}

	int innovation_number(0);
	for (int o_i(0); o_i < p_output_count; ++o_i) {
		for (int i_i(0); i_i < p_input_count; ++i_i) {

			add_link(
					i_i,
					p_input_count + o_i, // Output neurons are after inputs neurons
					p_randomize_weights ? Math::random(-1, 1) : 1,
					false /* Recurrent */,
					++innovation_number);
		}
	}
}

uint32_t brain::NtGenome::add_neuron(
		NtNeuronGene::NeuronGeneType p_type) {

	const uint32_t id(neuron_genes.size());

	neuron_genes.push_back(NtNeuronGene(
			id,
			p_type));
	return id;
}

uint32_t brain::NtGenome::add_link(
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id,
		real_t p_weight,
		bool p_recurrent,
		uint32_t p_innovation_number) {

	ERR_FAIL_INDEX_V(p_parent_neuron_id, neuron_genes.size(), 0);
	ERR_FAIL_INDEX_V(p_child_neuron_id, neuron_genes.size(), 0);
	ERR_FAIL_COND_V(0 > find_link(p_parent_neuron_id, p_child_neuron_id), 0);
	ERR_FAIL_COND_V(biggest_innovation_number >= p_innovation_number, 0);

	const uint32_t id(link_genes.size());

	link_genes.push_back(
			NtLinkGene(
					id,
					true,
					p_parent_neuron_id,
					p_child_neuron_id,
					p_weight,
					p_recurrent,
					p_innovation_number));

	neuron_genes[p_parent_neuron_id].outcoming_links.push_back(id);
	neuron_genes[p_child_neuron_id].incoming_links.push_back(id);

	biggest_innovation_number = p_innovation_number;

	return id;
}

uint32_t brain::NtGenome::get_link_count() const {
	return link_genes.size();
}

const brain::NtLinkGene *brain::NtGenome::get_link(int p_i) const {
	ERR_FAIL_INDEX_V(p_i, link_genes.size(), nullptr);
	return link_genes.data() + p_i;
}

void brain::NtGenome::active_link(uint32_t p_link_id) {
	ERR_FAIL_INDEX(p_link_id, link_genes.size());

	link_genes[p_link_id].active = true;
}

void brain::NtGenome::suppress_link(uint32_t p_link_id) {
	ERR_FAIL_INDEX(p_link_id, link_genes.size());

	link_genes[p_link_id].active = false;
}

uint32_t brain::NtGenome::find_link(
		uint32_t p_parent_neuron_id,
		uint32_t p_child_neuron_id) {

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		if (p_parent_neuron_id == it->parent_neuron_id)
			if (p_child_neuron_id == it->child_neuron_id)
				return it->id;
	}

	return -1;
}

void brain::NtGenome::mutate_all_link_weights(map_real_1 p_map_func) {

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		it->weight = p_map_func(it->weight);
	}
}

void brain::NtGenome::mutate_all_link_weights(map_real_2_ptr p_map_func, void *p_data) {

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		it->weight = p_map_func(it->weight, p_data);
	}
}

void brain::NtGenome::mutate_random_link_weight(map_real_2_ptr p_map_func, void *p_data) {

	ERR_FAIL_COND(!link_genes.size());

	NtLinkGene &lg =
			link_genes[static_cast<int>(Math::random(0, link_genes.size() - 1) + 0.5)];
	lg.weight = p_map_func(lg.weight, p_data);
}

void brain::NtGenome::mutate_random_link_toggle_activation() {

	ERR_FAIL_COND(!link_genes.size());

	NtLinkGene &lg =
			link_genes[static_cast<int>(Math::random(0, link_genes.size() - 1) + 0.5)];
	lg.active = !lg.active;
}

bool brain::NtGenome::mutate_add_random_link(
		real_t p_spawn_recurrent_threshold,
		std::vector<NtInnovation> &r_innovations,
		uint32_t &r_current_innovation_number) {

	const bool spawn_recurrent = Math::randd() < p_spawn_recurrent_threshold;
	const int max_tries(10);

	// Taking the non inputs to make it easier choose a non input neuron
	std::vector<uint32_t> non_input_neurons;
	non_input_neurons.reserve(neuron_genes.size());

	for (auto it = neuron_genes.begin(); it != neuron_genes.end(); ++it) {
		if (it->type != NtNeuronGene::NEURON_GENE_TYPE_INPUT)
			non_input_neurons.push_back(it->id);
	}

	const int non_input_neurons_last_index = non_input_neurons.size() - 1;
	const int neurons_last_index = neuron_genes.size() - 1;

	bool good_link = false;
	uint32_t parent_neuron_id;
	uint32_t child_neuron_id;

	for (int tries = 0; tries < max_tries; ++tries) {

		// Spawn a recurrent link
		if (spawn_recurrent && Math::randd() < 0.35) {
			// Spawn a self recurrent link
			/// Since a self recurrent can spawn by taking everything randomly
			/// a 35% of chance seems fine to me
			parent_neuron_id =
					non_input_neurons[(int)(Math::random(0, non_input_neurons_last_index) + 0.5f)];

			child_neuron_id = parent_neuron_id;

		} else {

			// Take everything randomly
			parent_neuron_id =
					neuron_genes[(int)(Math::random(0, neurons_last_index) + 0.5f)].id;

			child_neuron_id =
					non_input_neurons[(int)(Math::random(0, non_input_neurons_last_index) + 0.5f)];
		}

		if (-1 != find_link(parent_neuron_id, child_neuron_id))
			continue; // This link already exist

		const int should_be_recurrent =
				is_link_recurrent(parent_neuron_id, child_neuron_id);

		if (should_be_recurrent != spawn_recurrent)
			continue; // This link doesn't meets the requirements

		good_link = true;
		break;
	}

	if (!good_link)
		return false;

	// Search if this innovation already exist
	int innovation_index = find_innovation(
			r_innovations,
			NtInnovation::INNOVATION_LINK,
			parent_neuron_id,
			child_neuron_id,
			spawn_recurrent);

	uint32_t innovation_num;

	if (0 <= innovation_index) {
		// This mutation is not an innovation
		innovation_num = r_innovations[innovation_index].innovation_number;

	} else {

		// This is a novel innovation
		innovation_num = ++r_current_innovation_number;

		r_innovations.push_back(
				{ NtInnovation::INNOVATION_LINK,
						parent_neuron_id,
						child_neuron_id,
						spawn_recurrent,
						innovation_num });
	}

	add_link(
			parent_neuron_id,
			child_neuron_id,
			Math::random(-1, 1), // Weight <-- Just a random num
			spawn_recurrent,
			innovation_num);

	sort_genes();
	return true;
}

bool brain::NtGenome::mutate_add_random_neuron(
		std::vector<NtInnovation> &r_innovations,
		uint32_t &r_current_innovation_number) {

	/// Step 1. Find the link to split

	std::vector<int> active_links;
	active_links.reserve(link_genes.size());

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		if (!it->active)
			continue;

		active_links.push_back(it->id);
	}

	bool found = false;
	NtLinkGene link_to_split;

	if (active_links.size() < 15) {
		// When the genome is small, bias the search to the older connections

		for (int tries = 0; tries < 3 && !found; ++tries) {
			/// Iterate in reverse so all active older links will have more
			/// probability to get split
			for (auto it = active_links.rbegin(); it != active_links.rend(); ++it) {
				if (Math::randd() < 0.3f) {
					link_to_split = link_genes[*it];
					found = true;
					break;
				}
			}
		}
	} else {

		// Take one random link with normal distribution
		const int link_id = active_links[static_cast<int>(
				Math::random(0.f, active_links.size() - 1.f) + 0.5f)];

		link_to_split = link_genes[link_id];
		found = true;
	}

	if (!found)
		return false; // No link to split

	int innovation_index = find_innovation(
			r_innovations,
			NtInnovation::INNOVATION_NODE,
			link_to_split.parent_neuron_id,
			link_to_split.child_neuron_id,
			false);

	uint32_t in_link_innovation_number;
	uint32_t out_link_innovation_number;

	if (0 <= innovation_index) {
		in_link_innovation_number = r_innovations[innovation_index].innovation_number;
		out_link_innovation_number = in_link_innovation_number + 1;
	} else {
		// Novel innovation
		in_link_innovation_number = ++r_current_innovation_number;
		out_link_innovation_number = ++r_current_innovation_number;

		r_innovations.push_back(
				{ NtInnovation::INNOVATION_NODE,
						link_to_split.parent_neuron_id,
						link_to_split.child_neuron_id,
						false, // <-- doesn't matter in this case
						in_link_innovation_number });
	}

	suppress_link(link_to_split.id);
	const uint32_t new_neuron_id = add_neuron(NtNeuronGene::NEURON_GENE_TYPE_HIDDEN);

	add_link(
			link_to_split.parent_neuron_id,
			new_neuron_id,
			1.f,
			link_to_split.recurrent,
			in_link_innovation_number);

	add_link(
			new_neuron_id,
			link_to_split.child_neuron_id,
			link_to_split.weight,
			false,
			out_link_innovation_number);

	sort_genes();
	return true;
}

bool brain::NtGenome::mate_multipoint(
		const NtGenome &p_mom,
		real_t p_mom_fitness,
		const NtGenome &p_daddy,
		real_t p_daddy_fitness,
		bool p_average) {

	neuron_genes.clear();
	link_genes.clear();

	const NtGenome *innovative = &p_mom; // Most innovated
	const NtGenome *obsolete = &p_daddy; // less_innovated
	bool is_innovative_fitter = p_mom_fitness > p_daddy_fitness;

	if (innovative->get_innovation_number() < obsolete->get_innovation_number()) {
		innovative = &p_daddy;
		obsolete = &p_mom;
		is_innovative_fitter = !is_innovative_fitter;
	}

	// Add all the neurons of the best genome in the same order
	if (is_innovative_fitter) {
		for (auto it = innovative->neuron_genes.begin(); it != innovative->neuron_genes.end(); ++it) {
			add_neuron(it->type);
		}
	} else {
		for (auto it = obsolete->neuron_genes.begin(); it != obsolete->neuron_genes.end(); ++it) {
			add_neuron(it->type);
		}
	}

	const int bigger_in_num = innovative->get_innovation_number();
	auto it_inn = innovative->link_genes.begin();
	auto it_obs = obsolete->link_genes.begin();

	for (int i = 0; i <= bigger_in_num; ++i) {

		auto genome_inn = innovative->link_genes.end();
		auto genome_obs = obsolete->link_genes.end();

		if (it_inn != innovative->link_genes.end() && it_inn->innovation_number == i) {
			genome_inn = it_inn++;
		}
		if (it_obs != obsolete->link_genes.end() && it_obs->innovation_number == i) {
			genome_obs = it_obs++;
		}

		bool want_to_add = false;
		NtLinkGene gene_to_add;

		if (genome_inn != innovative->link_genes.end() &&
				genome_obs != obsolete->link_genes.end()) {

			if (p_average) {
				// Both have this innovation, average them

				gene_to_add = *genome_inn;
				gene_to_add.weight = (genome_inn->weight + genome_obs->weight) * 0.5;

				if (Math::randd() < 0.5) {
					gene_to_add.active = genome_inn->active;
				} else {
					gene_to_add.active = genome_obs->active;
				}

			} else {
				// Both have this innovation, select one randomly
				if (Math::randd() < 0.5) {
					gene_to_add = *genome_inn;
				} else {
					gene_to_add = *genome_obs;
				}
			}
			want_to_add = true;

		} else if (genome_inn != innovative->link_genes.end()) {
			// Only the bigger genome has this innovation

			if (is_innovative_fitter) {

				gene_to_add = *genome_inn;
				want_to_add = true;
			}

		} else if (genome_obs != obsolete->link_genes.end()) {
			// Only the smaller genome has this innovation

			if (!is_innovative_fitter) {

				gene_to_add = *genome_obs;
				want_to_add = true;
			}

		} else {
			// Nobody have this genome
		}

		if (!want_to_add)
			continue;

		uint32_t id = add_link(
				gene_to_add.parent_neuron_id,
				gene_to_add.child_neuron_id,
				gene_to_add.weight,
				gene_to_add.recurrent,
				gene_to_add.innovation_number);

		if (!gene_to_add.active) {
			suppress_link(id);
		}
	}
}

bool brain::NtGenome::mate_singlepoint(
		const NtGenome &p_mom,
		const NtGenome &p_daddy) {

	neuron_genes.clear();
	link_genes.clear();

	const NtGenome *bigger = &p_mom;
	const NtGenome *smaller = &p_daddy;

	if (p_mom.get_link_count() < p_daddy.get_link_count()) {
		bigger = &p_daddy;
		smaller = &p_mom;
	}

	// Just copy all neurons of the bigger genome since at the end we will take
	// all links from the bigger one and fore sure will be used all neurons
	for (auto it = bigger->neuron_genes.begin(); it != bigger->neuron_genes.end(); ++it) {
		add_neuron(it->type);
	}

	const int cross_gene_small = static_cast<int>(Math::random(0, smaller->get_link_count() - 1) + 0.5);

	// Find the cut on the bigger genome by checking the innovation number in order to
	// avoid adding two times a gene
	int cross_gene_big(0);
	bool has_bigger_innov_num(false);
	const uint32_t innovation_number_to_search =
			smaller->get_link(cross_gene_small)->innovation_number;

	for (auto it = bigger->link_genes.begin(); it != bigger->link_genes.end(); ++it) {
		if (innovation_number_to_search == it->innovation_number) {
			has_bigger_innov_num = true;
			break;
		} else if (innovation_number_to_search < it->innovation_number) {
			// The innovation number is not in this genome
			// So decrease and make sure that this gene can be taken
			// during the split
			--cross_gene_big;
			has_bigger_innov_num = true;
			break;
		}
		++cross_gene_big;
	}

	ERR_FAIL_COND_V(!has_bigger_innov_num, false);

	// Copy genes from the smaller genome
	for (int i(0); i < cross_gene_small; ++i) {
		const NtLinkGene &link = smaller->link_genes[i];
		uint32_t id = add_link(
				link.parent_neuron_id,
				link.child_neuron_id,
				link.weight,
				link.recurrent,
				link.innovation_number);

		if (!link.active) {
			suppress_link(id);
		}
	}

	{
		NtLinkGene link = smaller->link_genes[cross_gene_small];
		const NtLinkGene &s = bigger->link_genes[cross_gene_big];

		if (link.innovation_number == s.innovation_number) {
			// Average the cross gene from the two genome

			link.weight += s.weight;
			link.weight *= 0.5;

			if (link.active != s.active) {
				link.active = Math::randd() < 0.5f;
			}

			uint32_t id = add_link(
					link.parent_neuron_id,
					link.child_neuron_id,
					link.weight,
					link.recurrent,
					link.innovation_number);

			if (!link.active) {
				suppress_link(id);
			}

		} else {

			// The innovation number is not found so just take the small

			uint32_t id = add_link(
					link.parent_neuron_id,
					link.child_neuron_id,
					link.weight,
					link.recurrent,
					link.innovation_number);

			if (!link.active) {
				suppress_link(id);
			}
		}
	}

	// Copy the genes from the bigger genome
	for (int i(cross_gene_big + 1); i < bigger->get_link_count(); ++i) {
		const NtLinkGene &link = bigger->link_genes[i];
		uint32_t id = add_link(
				link.parent_neuron_id,
				link.child_neuron_id,
				link.weight,
				link.recurrent,
				link.innovation_number);

		if (!link.active) {
			suppress_link(id);
		}
	}

	return true;
}

void brain::NtGenome::generate_neural_network(SharpBrainArea &r_brain_area) const {

	r_brain_area.clear();

	for (auto it = neuron_genes.begin(); it != neuron_genes.end(); ++it) {

		const uint32_t id = r_brain_area.add_neuron();

		ERR_FAIL_COND(id != it->id);

		switch (it->type) {
			case NtNeuronGene::NEURON_GENE_TYPE_INPUT:
				r_brain_area.set_neuron_as_input(id);
				break;
			case NtNeuronGene::NEURON_GENE_TYPE_OUTPUT:
				r_brain_area.set_neuron_as_output(id);
				break;
		}
	}

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		if (!it->active)
			continue;

		r_brain_area.add_link(
				it->parent_neuron_id,
				it->child_neuron_id,
				it->weight,
				it->recurrent);
	}
}

void brain::NtGenome::clear() {
	neuron_genes.clear();
	link_genes.clear();
}

void brain::NtGenome::duplicate_in(NtGenome &p_genome) const {

	p_genome.clear();

	p_genome.neuron_genes.reserve(neuron_genes.size());
	p_genome.link_genes.reserve(link_genes.size());

	// TODO try to use std::copy instead

	for (auto it = neuron_genes.begin(); it != neuron_genes.end(); ++it) {

		p_genome.neuron_genes.push_back(*it);
	}

	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {

		p_genome.link_genes.push_back(*it);
	}
}

void brain::NtGenome::sort_genes() {

	std::sort(link_genes.begin(), link_genes.end(), gene_innovation_comparator);

	// Update the ids and creates the map of old ids

	std::vector<int> id_map;
	id_map.resize(link_genes.size());

	int new_id(0);
	for (auto it = link_genes.begin(); it != link_genes.end(); ++it) {
		id_map[it->id] = new_id;
		it->id = new_id;
		++new_id;
	}

	// Now update the neuron links
	for (auto it = neuron_genes.begin(); it != neuron_genes.end(); ++it) {
		for (auto it_inc = it->incoming_links.begin(); it_inc != it->incoming_links.end(); ++it_inc) {
			*it_inc = id_map[*it_inc];
		}
		for (auto it_out = it->outcoming_links.begin(); it_out != it->outcoming_links.end(); ++it_out) {
			*it_out = id_map[*it_out];
		}
	}
}

bool brain::NtGenome::check_innovation_numbers() const {
	if (1 >= link_genes.size()) {
		return true;
	}

	uint32_t increment(link_genes.begin()->innovation_number);

	for (auto it = link_genes.begin() + 1; it != link_genes.end(); ++it) {
		if (increment < it->innovation_number) {
			increment = it->innovation_number;
		} else {
			return false;
		}
	}
	return true;
}

uint32_t brain::NtGenome::get_innovation_number() const {
	return biggest_innovation_number;
}

bool brain::NtGenome::is_link_recurrent(
		NeuronId p_parent_neuron_id,
		NeuronId p_child_neuron_id) const {

	ERR_FAIL_INDEX_V(p_parent_neuron_id, neuron_genes.size(), false);
	ERR_FAIL_INDEX_V(p_child_neuron_id, neuron_genes.size(), false);

	if (p_parent_neuron_id == p_child_neuron_id)
		return true; // Linkage to itself is recurrent

	// Try to find the p_child_neuron_id in backward (From incoming) to see if
	// there's a dependency
	const std::vector<int> &parent_inc = neuron_genes[p_parent_neuron_id].incoming_links;
	for (auto it = parent_inc.begin(); it != parent_inc.end(); ++it) {

		const int link_id = *it;

		// Skip if recurrent
		if (link_genes[link_id].recurrent)
			continue;

		if (_recursive_is_link_recurrent(
					p_parent_neuron_id,
					link_genes[link_id].parent_neuron_id,
					p_child_neuron_id))
			return true;
	}

	return false;
}

bool brain::NtGenome::_recursive_is_link_recurrent(
		NeuronId p_parent_neuron_id,
		NeuronId p_middle_neuron_id,
		NeuronId p_child_neuron_id) const {

	if (p_parent_neuron_id == p_middle_neuron_id) {
		// When this happen a connection loop is detected, so
		return true;
	}

	if (p_middle_neuron_id == p_child_neuron_id)
		return true; // Recursion found

	// Try to find the p_child_neuron_id in backward (From incoming) to see if
	// there's a dependency
	const std::vector<int> &parent_inc =
			neuron_genes[p_middle_neuron_id].incoming_links;
	for (auto it = parent_inc.begin(); it != parent_inc.end(); ++it) {

		const int link_id = *it;

		// Skip if recurrent
		if (link_genes[link_id].recurrent)
			continue;

		if (_recursive_is_link_recurrent(
					p_parent_neuron_id,
					link_genes[link_id].parent_neuron_id,
					p_child_neuron_id))
			return true;
	}

	return false;
}

int brain::NtGenome::find_innovation(
		std::vector<NtInnovation> &p_innovations,
		NtInnovation::InnovationType p_innovation_type,
		NeuronId p_parent_neuron_id,
		NeuronId p_child_neuron_id,
		bool p_is_recurrent) {

	for (auto it = p_innovations.begin(); it != p_innovations.end(); ++it) {

		if (it->type != p_innovation_type)
			continue;

		if (it->parent_neuron_id != p_parent_neuron_id)
			continue;

		if (it->child_neuron_id != p_child_neuron_id)
			continue;

		if (p_innovation_type == NtInnovation::INNOVATION_LINK) {
			if (it->is_recurrent != p_is_recurrent)
				continue;
		}

		return it - p_innovations.begin();
	}

	return -1;
}

bool gene_innovation_comparator(brain::NtLinkGene &p_1, brain::NtLinkGene &p_2) {
	return p_1.innovation_number < p_2.innovation_number;
}
