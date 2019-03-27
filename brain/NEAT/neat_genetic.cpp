#include "neat_genetic.h"

#include "brain/NEAT/neat_genome.h"
#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"

real_t brain::NtGenetic::compatibility(
		const NtGenome &p_genome_1,
		const NtGenome &p_genome_2,
		real_t p_disjoints_significance,
		real_t p_excesses_significance,
		real_t p_weights_significance) {

	if (!p_genome_1.get_link_count()) // TODO remove this
		ERR_FAIL_COND_V(!p_genome_1.get_link_count(), -1);
	ERR_FAIL_COND_V(!p_genome_2.get_link_count(), -1);

	real_t D(0);
	real_t E(0);

	// Find smallest and bigger genome

	const uint32_t biggest_innovation(
			MAX(p_genome_1.get_innovation_number(),
					p_genome_2.get_innovation_number()));

	// Count disjoints, Count excesses by checking if both have the
	// innovation number
	real_t g1_weights_sum(0.f);
	real_t g2_weights_sum(0.f);

	for (
			int innovation(0), g1_i(0), g2_i(0);
			innovation <= biggest_innovation;
			++innovation) {

		bool g1_has_innovation(false);
		bool g2_has_innovation(false);
		bool someone_is_over(false);

		if (g1_i < p_genome_1.get_link_count()) {
			const NtLinkGene *link = p_genome_1.get_link(g1_i);
			if (innovation == link->innovation_number) {
				g1_has_innovation = true;
				g1_weights_sum += link->weight;
				++g1_i;
			}
		} else {
			someone_is_over = true;
		}

		if (g2_i < p_genome_2.get_link_count()) {
			const NtLinkGene *link = p_genome_2.get_link(g2_i);
			if (innovation == link->innovation_number) {
				g2_has_innovation = true;
				g2_weights_sum += link->weight;
				++g2_i;
			}
		} else {
			someone_is_over = true;
		}

		if (g1_has_innovation != g2_has_innovation) {

			if (!someone_is_over) {
				// This discrepancy is a disjoint
				D += 1;
			} else {
				// This discrepancy is a excess
				E += 1;
			}
		}
	}

	real_t W = Math::abs(Math::abs((g1_weights_sum / p_genome_1.get_link_count())) -
						 Math::abs((g2_weights_sum / p_genome_2.get_link_count())));

	/// The research says that for smaller genome the normalization is not necessary
	/// and can be set 1.
	/// Setting N=1 gives the possibility to controll the compatibility
	/// in a better way.
	/// For this reason I'm putting 1 even for bigger genome
	// Count genes of the bigger genome
	//real_t N(MAX(p_genome_1.get_link_count(), p_genome_2.get_link_count()));
	real_t N(1);

	const real_t cD(p_disjoints_significance);
	const real_t cE(p_excesses_significance);
	const real_t cW(p_weights_significance);

	return ((cD * D) / N) + ((cE * E) / N) + (cW * W);
}
