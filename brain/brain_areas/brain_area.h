#pragma once

#include "brain/math/matrix.h"

namespace brain {

enum BrainAreaType {
	BRAIN_AREA_TYPE_UNIFORM,
	BRAIN_AREA_TYPE_NEAT
};

/**
 * @brief The BrainArea class
 *
 * This class is the base type for each type of Neural Network
 * The idea is to give the possibility to link more brain areas to create
 * one single brain that takes more complex decision.
 */
class BrainArea {

	/**
	 * @brief type
	 *
	 * The type of this brain area
	 */
	BrainAreaType type;

public:
	BrainArea(BrainAreaType p_type);

	/**
	 * @brief get_type
	 * @return
	 *
	 * Return the type of this brain area
	 */
	BrainAreaType get_type() const;

	/**
	 * @brief randomize_weights randomize the weights
	 * between the passed -range and range
	 *
	 * @param p_range
	 */
	virtual void randomize_weights(real_t p_range) = 0;

	/**
	 * @brief fill_weights with the passed value
	 * @param p_value
	 */
	virtual void fill_weights(real_t p_value) = 0;

	/**
	 * @brief get_input_layer_size
	 * @return
	 *
	 * Get the input layer size
	 */
	virtual uint32_t get_input_layer_size() const = 0;

	/**
	 * @brief get_output_layer_size
	 * @return
	 *
	 * Get the output layer size
	 */
	virtual uint32_t get_output_layer_size() const = 0;

	/**
	 * @brief guess
	 * @param p_input Input data
	 * @param r_guess result
	 *
	 * Make a guess considering the inputs
	*/
	virtual void guess(
			const Matrix &p_input,
			Matrix &r_guess) const = 0;
};

} // namespace brain
