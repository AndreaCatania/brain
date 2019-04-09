#pragma once

#include "brain/math/math_defs.h"

namespace brain {

class NtGenome;

/**
 * @brief The NtGenetic class is a collection of functions to perform genetic
 * operations
 */
class NtGenetic {
public:
	// TODO think about port this inside the Genome as static function
	/**
	 * @brief compatibility function returns a crescent value more the difference
	 * between the genomes increase.
	 *
	 * To calculate this value is used this linear equation:
	 *
	 * Δ = ((cD * D) / N) + ((cE * E) / N) + (cW * W)
	 *
	 * Δ is the result delta difference
	 * D is the number of disjoints genes
	 * E is the number of the excesses genes
	 * W is the weights average difference of two genomes
	 * N genes count of bigger genome
	 * cD is the Fisjoints significance coefficient
	 * cE is the Excesses significance coefficient
	 * cW is the Weights significance coefficient
	 *
	 * Compatibility of Genome1 and Genome2
	 * [gene1][gene2][gene3][gene4]       [gene6]
	 * [gene1][gene2]              [gene5]
	 *
	 * The disjoint genes are that genes that are missing in the other genome
	 * For example the gene3, gene4, gene6.
	 *
	 * The excesses genes are that overflow the other genome, like the gene6
	 *
	 * By chaning the coefficients is possible to control the amount of importance
	 * of each part of the equation.
	 *
	 * For example a p_weights_significance of 0 make sure that until the topology
	 * is changed the result is 0.
	 *
	 * @param p_genome_1
	 * @param p_genome_2
	 * @param p_disjoints_significance
	 * @param p_excesses_significance
	 * @param p_weights_significance
	 * @return
	 */
	static real_t compatibility(
			const NtGenome &p_genome_1,
			const NtGenome &p_genome_2,
			real_t p_disjoints_significance,
			real_t p_excesses_significance,
			real_t p_weights_significance);
};

} // namespace brain
