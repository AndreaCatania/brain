

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

void test_uniform_ba_XOR() {

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
		std::shuffle(
				inputs.begin(),
				inputs.end(),
				std::default_random_engine(seed));

		std::shuffle(
				expected.begin(),
				expected.end(),
				std::default_random_engine(seed));
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

#include "brain/NEAT/neat_genetic.h"
#include "brain/NEAT/neat_population.h"
#include "brain/brain_areas/sharp_brain_area.h"

const uint32_t iterations = 1;

void test_NEAT_XOR() {

	std::vector<brain::Matrix> inputs;
	std::vector<brain::Matrix> expected;
	{
		real_t a[] = { 1, 1, 0 };
		real_t b[] = { 1 };
		inputs.push_back(brain::Matrix(3, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}
	{
		real_t a[] = { 1, 0, 1 };
		real_t b[] = { 1 };
		inputs.push_back(brain::Matrix(3, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}
	{
		real_t a[] = { 1, 1, 1 };
		real_t b[] = { 0 };
		inputs.push_back(brain::Matrix(3, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}
	{
		real_t a[] = { 1, 0, 0 };
		real_t b[] = { 0 };
		inputs.push_back(brain::Matrix(3, 1, a));
		expected.push_back(brain::Matrix(1, 1, b));
	}

	brain::NtPopulationSettings settings;
	settings.seed = time(nullptr);

	/// Step 1. Population creation
	brain::NtPopulation population(
			brain::NtGenome(3, 1, true),
			100 /*population size*/,
			settings);

	const int epoch_max(20);
	for (int epoch(0); epoch < epoch_max; ++epoch) {

		/// Step 2. Population testing and evaluation
		for (int i = population.get_organism_count() - 1; 0 <= i; --i) {
			const brain::SharpBrainArea *brain_area = population.organism_get_network(i);
			brain::Matrix result;
			for (int k(inputs.size() - 1); 0 <= k; --k) {
				brain_area->guess(inputs[k], result);
				real_t error = result.get(0, 0) - expected[k].get(0, 0);
				population.organism_add_fitness(i, 1.f - ABS(error));
			}
		}

		/// Step 3. advance the epoch
		population.epoch_advance();
	}

	// TODO get the population champion and test it.
	int b = 0;
}

int main() {

	brain::ErrorHandlerList *error_handler = new brain::ErrorHandlerList;
	error_handler->errfunc = print_error_callback;
	brain::add_error_handler(error_handler);

	test_NEAT_XOR();

	//test_uniform_ba_XOR();

	return 0;
}
