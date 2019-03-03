#include "brain_area.h"
#include "brain/math/math_funcs.h"

activation_func brain::BrainArea::activation_functions[] = { brain::Math::sigmoid };
activation_func brain::BrainArea::activation_derivatives[] = { brain::Math::sigmoid_fast_derivative };

brain::BrainArea::BrainArea(BrainAreaType p_type) :
		type(p_type) {
}
