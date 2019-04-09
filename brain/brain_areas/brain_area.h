#pragma once

#include "brain/math/matrix.h"

typedef real_t (*activation_func)(real_t p_val);

namespace brain {

/**
 * @brief The BrainAreaType enum is used to point the type of brain area
 */
enum BrainAreaType {
	BRAIN_AREA_TYPE_UNIFORM,
	BRAIN_AREA_TYPE_SHARP
};

/**
 * @brief The BrainArea class
 *
 * This class is the base type for each type of Neural Network
 * The idea is to give the possibility to link more brain areas to create
 * one single brain that takes more complex decision.
 */
class BrainArea {

public:
	/**
	 * @brief The Activation enum is used to indicate the type of
	 * the activation func
	 */
	enum Activation {
		ACTIVATION_SIGMOID,
		ACTIVATION_RELU,
		ACTIVATION_LEAKY_RELU,
		ACTIVATION_TANH,
		ACTIVATION_LINEAR,
		ACTIVATION_BINARY,

		// Special activation functions
		ACTIVATION_SOFTMAX,

		ACTIVATION_MAX
	};

	/**
	 * @brief activation_functions is a vector that holds the activations
	 * ordered by Activation ID
	 */
	static activation_func activation_functions[];

	/**
	 * @brief activation_derivatives is a vector that holds the derivatives
	 * ordered by Activation ID
	 */
	static activation_func activation_derivatives[];

private:
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
	virtual bool guess(
			const Matrix &p_input,
			Matrix &r_guess) const = 0;
};

} // namespace brain
