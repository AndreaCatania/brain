

#include "brain/brain_areas/uniform_brain_area.h"
#include "brain/error_handler.h"
#include "brain/math/math_funcs.h"
#include "brain/math/matrix.h"
#include "brain/string.h"
#include "brain/typedefs.h"
#include <time.h>
#include <algorithm>
#include <random>

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

void test_brain_area_train(brain::UniformBrainArea &area) {

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
	brain::UniformBrainArea area1;
	area1.set_input_layer_size(4);
	area1.set_output_layer_size(4);
	area1.set_hidden_layers_count(1);
	area1.set_hidden_layer(0, 4, brain::UniformBrainArea::ACTIVATION_SIGMOID);
	area1.randomize_weights(2);
	area1.fill_biases(1.0);

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
	brain::UniformBrainArea area1(2, 1, 1);

	area1.set_hidden_layer(0, 2, brain::UniformBrainArea::ACTIVATION_SIGMOID);

	brain::Math::seed(time(nullptr));
	area1.randomize_weights(1);
	area1.randomize_biases(1);

	std::vector<brain::Matrix> inputs;
	std::vector<brain::Matrix> expected;
	{
		real_t a[] = { 1, 0 };
		real_t b[] = { 1 };
		inputs.push_back(brain::Matrix(2, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}
	{
		real_t a[] = { 0, 1 };
		real_t b[] = { 1 };
		inputs.push_back(brain::Matrix(2, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}
	{
		real_t a[] = { 1, 1 };
		real_t b[] = { 0 };
		inputs.push_back(brain::Matrix(2, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}
	{
		real_t a[] = { 0, 0 };
		real_t b[] = { 0 };
		inputs.push_back(brain::Matrix(2, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}

	real_t error;
	brain::UniformBrainArea::LearningCache lc;
	for (int t(0); t < 100000; ++t) {
		for (int i(0); i < inputs.size(); ++i) {
			error = area1.learn(inputs[i], expected[i], 0.05, &lc);
		}

		// SGD prefer shuffled data
		uint32_t seed = time(nullptr);
		std::shuffle(inputs.begin(), inputs.end(), std::default_random_engine(seed));
		std::shuffle(expected.begin(), expected.end(), std::default_random_engine(seed));
	}

	print_line("Error: " + brain::rtos(error));

	// Just guess now
	brain::Matrix guess;

	{
		real_t x[] = { 1, 0 };
		brain::Matrix input(2, 1, x);
		area1.guess(input, guess);
		print_line(std::string(input) + " Guess: " + std::string(guess));
	}
	{
		real_t x[] = { 1, 1 };
		brain::Matrix input(2, 1, x);
		area1.guess(input, guess);
		print_line(std::string(input) + " Guess: " + std::string(guess));
	}
	{
		real_t x[] = { 0, 1 };
		brain::Matrix input(2, 1, x);
		area1.guess(input, guess);
		print_line(std::string(input) + " Guess: " + std::string(guess));
	}
	{
		real_t x[] = { 0, 0 };
		brain::Matrix input(2, 1, x);
		area1.guess(input, guess);
		print_line(std::string(input) + " Guess: " + std::string(guess));
	}
}

#include "brain/brain_areas/sharp_brain_area.h"
void test_NEAT_XOR() {

	brain::SharpBrainArea brain_area;
}

int main() {

	brain::ErrorHandlerList *error_handler = new brain::ErrorHandlerList;
	error_handler->errfunc = print_error_callback;
	brain::add_error_handler(error_handler);

	//test_brain_area();
	test_NEAT_XOR();

	return 0;
}
