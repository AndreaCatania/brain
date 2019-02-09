

#include "brain/areas/brain_area.h"
#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include "brain/math/matrix.h"
#include "brain/string.h"
#include "brain/typedefs.h"
#include <time.h>

void print_line(const std::string &p_msg) {
	printf("[INFO] ");
	printf(p_msg.c_str());
	printf("\n");
}

void print_error_callback(
		void *p_user_data,
		const char *p_function,
		const char *p_file,
		int p_line,
		const char *p_error,
		const char *p_explain,
		brain::ErrorHandlerType p_type) {

	std::string msg =
			std::string() +
			(p_type == brain::ERR_HANDLER_ERROR ? "[ERROR] " : "[WARN]") +
			p_file +
			" Function: " + p_function +
			", line: " +
			brain::itos(p_line) +
			"\n\t" + p_error +
			" " + p_explain;

	printf(msg.c_str());
	printf("\n");
}

void test_brain_area_train(brain::BrainArea &area) {

	std::vector<brain::Matrix> inputs;
	std::vector<brain::Matrix> expected;
	{
		real_t a[] = { 1, 0, 0, 0 };
		real_t b[] = { 1, 0, 0, 0 };
		inputs.push_back(brain::Matrix(4, 1, a));
		expected.push_back(brain::Matrix(4, 1, b));
	}
	{
		real_t a[] = { 0, 1, 0, 0 };
		real_t b[] = { 0, 1, 0, 0 };
		inputs.push_back(brain::Matrix(4, 1, a));
		expected.push_back(brain::Matrix(4, 1, b));
	}
	{
		real_t a[] = { 0, 0, 1, 0 };
		real_t b[] = { 0, 0, 1, 0 };
		inputs.push_back(brain::Matrix(4, 1, a));
		expected.push_back(brain::Matrix(4, 1, b));
	}
	{
		real_t a[] = { 0, 0, 0, 1 };
		real_t b[] = { 0, 0, 0, 1 };
		inputs.push_back(brain::Matrix(4, 1, a));
		expected.push_back(brain::Matrix(4, 1, b));
	}

	for (int i(0); i < inputs.size(); ++i) {

		const real_t accuracy = area.learn(inputs[i], expected[i], 0.1, nullptr);
		print_line("Error: " + brain::rtos(accuracy));
	}
}

void test_complex_brain_area() {

	// Create brain area
	brain::BrainArea area1;
	area1.set_input_layer_size(4);
	area1.set_output_layer_size(4);
	area1.resize_hidden_layers(1);
	area1.set_hidden_layer(0, 4, brain::BrainArea::ACTIVATION_SIGMOID);
	area1.randomize_weights(2);
	area1.set_biases(1.0);

	test_brain_area_train(area1);

	real_t features_m[] = { 1, 0, 0, 0 };
	brain::Matrix features(4, 1, features_m);

	brain::Matrix guess;
	area1.guess(features, guess);

	print_line("Result:");
	print_line(guess);
}

void test_brain_area() {

	// Create brain area
	brain::BrainArea area1;

	area1.set_input_layer_size(2);

	area1.resize_hidden_layers(1);
	area1.set_hidden_layer(0, 2, brain::BrainArea::ACTIVATION_SIGMOID);

	area1.set_output_layer_size(2);

	brain::Math::seed(time(nullptr));
	area1.randomize_weights(1);
	area1.randomize_biases(1);

	real_t inputs[] = { 1, 0 };
	brain::Matrix input(2, 1, inputs);

	real_t expected_m[] = { 1, 0 };
	brain::Matrix expected(2, 1, expected_m);

	real_t error;
	brain::BrainArea::LearningCache lc;
	for (int i(0); i < 10000; ++i) {
		error = area1.learn(input, expected, 0.05, &lc);
	}

	// Just guess
	brain::Matrix guess;
	area1.guess(input, guess);

	print_line("Error: " + brain::rtos(error) + " result " + std::string(guess));
}

int main() {

	brain::ErrorHandlerList *error_handler = new brain::ErrorHandlerList;
	error_handler->errfunc = print_error_callback;
	brain::add_error_handler(error_handler);

	test_brain_area();

	return 0;
}
