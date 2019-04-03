#include "brain_area.h"
#include "brain/math/math_funcs.h"

activation_func brain::BrainArea::activation_functions[] = {
	brain::Math::sigmoid,
	brain::Math::relu,
	brain::Math::leaky_relu,
	brain::Math::tanh,
};

activation_func brain::BrainArea::activation_derivatives[] = {
	brain::Math::sigmoid_fast_derivative,
	brain::Math::relu_derivative,
	brain::Math::leaky_relu_derivative,
	brain::Math::tanh_derivative,
};

brain::BrainArea::BrainArea(BrainAreaType p_type) :
		type(p_type) {
}
