#include "brain_area.h"
#include "brain/math/math_funcs.h"

activation_func brain::BrainArea::activation_functions[] = {
	brain::Math::sigmoid,
	brain::Math::relu,
	brain::Math::leaky_relu,
	brain::Math::tanh,
	brain::Math::linear,
	brain::Math::binary_step,
	brain::Math::soft_max_allert
};

activation_func brain::BrainArea::activation_derivatives[] = {
	brain::Math::sigmoid_derivative,
	brain::Math::relu_derivative,
	brain::Math::leaky_relu_derivative,
	brain::Math::tanh_derivative,
	brain::Math::linear_derivative,
	brain::Math::binary_step_derivative,
	brain::Math::soft_max_derivative
};

brain::BrainArea::BrainArea(BrainAreaType p_type) :
		type(p_type) {
}
