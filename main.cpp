

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
	fflush(stdout);
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
	fflush(stdout);
}

void test_uniform_ba_XOR() {

	// Create brain area
	brain::UniformBrainArea area1(2, 1, 1);

	area1.set_hidden_layer(0, 2, brain::UniformBrainArea::ACTIVATION_LEAKY_RELU);
	area1.set_layer_activation(2, brain::UniformBrainArea::ACTIVATION_LINEAR);

	brain::Math::randomize();
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
	brain::UniformBrainArea::LearningData lc;

	int learn_mode = 0;
	if (learn_mode == 0) {

		// Online gradient descent
		for (int t(0); t < 100000; ++t) {

			for (int i(0); i < inputs.size(); ++i) {
				error = area1.learn(
						inputs[i],
						expected[i],
						0.1,
						true,
						NULL,
						&lc);
			}

			// GD prefer shuffled data
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

	} else if (learn_mode == 1) {

		// Batch gradient descent
		int samples_count = 0;
		brain::UniformBrainArea::DeltaGradients total_dg;
		brain::UniformBrainArea::DeltaGradients _volatile_dg;
		for (int t(0); t < 100000; ++t) {

			for (int i(0); i < inputs.size(); ++i) {
				error = area1.learn(
						inputs[i],
						expected[i],
						0.01,
						false,
						&_volatile_dg,
						&lc);

				++samples_count;
				total_dg += _volatile_dg;
			}

			// GD prefer shuffled data
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

		total_dg /= samples_count;

		area1.update_weights(total_dg);

	} else if (learn_mode == 2) {

		// Mini batch gradient descent
		int samples_count = 0;
		brain::UniformBrainArea::DeltaGradients total_dg;
		brain::UniformBrainArea::DeltaGradients _volatile_dg;
		for (int t(0); t < 100000; ++t) {

			for (int i(0); i < inputs.size(); ++i) {
				error = area1.learn(
						inputs[i],
						expected[i],
						0.01,
						false,
						&_volatile_dg,
						&lc);

				++samples_count;
				total_dg += _volatile_dg;
			}

			if (samples_count >= 64) {
				total_dg /= samples_count;
				area1.update_weights(total_dg);

				total_dg.weights.resize(0);
				total_dg.biases.resize(0);
				samples_count = 0;
			}

			// GD prefer shuffled data
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

		if (samples_count > 0) {
			total_dg /= samples_count;
			area1.update_weights(total_dg);
		}
	}

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

	/// Prepare datas
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

	const int epoch_max(100);

	// Statistics
	std::vector<brain::NtEpochStatistics> statistics;
	statistics.reserve(epoch_max);

	// Settings
	brain::NtPopulationSettings settings;
	settings.seed = time(nullptr);
	settings.seed = 1554825747; // TODO test seed

	brain::Math::seed(settings.seed);
	uint32_t shuffle_seed = settings.seed;

	// Population creation
	brain::NtPopulation population(
			brain::NtGenome(
					3,
					1,
					true,
					brain::BrainArea::ACTIVATION_RELU,
					brain::BrainArea::ACTIVATION_BINARY),
			150 /*population size*/,
			settings);

	// Execution
	for (int epoch(0); epoch < epoch_max; ++epoch) {

		// Shuffle is required to avoid create a pattern
		{
			std::shuffle(
					inputs.begin(),
					inputs.end(),
					// Re init with the same seed to maintain the order of shuffling
					std::default_random_engine(shuffle_seed + epoch));

			std::shuffle(
					expected.begin(),
					expected.end(),
					// Re init with the same seed to maintain the order of shuffling
					std::default_random_engine(shuffle_seed + epoch));
		}

		/// Step 2. Population testing and evaluation
		for (int i = 0; i < population.get_population_size(); ++i) {
			const brain::SharpBrainArea *brain_area = population.organism_get_network(i);
			brain::Matrix result;
			int acceptable_result(0);
			real_t total_error(0);
			for (int k(0); k < inputs.size(); ++k) {
				if (brain_area->guess(inputs[k], result)) {

					real_t error = result.get(0, 0) - expected[k].get(0, 0);
					error = ABS(error);
					if (error < 0.49f) {
						++acceptable_result;
					}
					total_error += error;
				} else {
					// If the guess function fail, put max error in order to avoid
					// that this organism can reproduce
					total_error += 1;
				}
			}

			real_t fitness = 1.f - (total_error / inputs.size());

			//const real_t brain_size_penality = 0.1 * brain_area->get_neuron_count();
			//fitness -= brain_size_penality;

			if (acceptable_result != inputs.size()) {

				population.organism_set_fitness(i, fitness);
			} else {

				// Give a bonus to all organisms that are able to guess
				// all situations
				population.organism_set_fitness(i, brain::Math::pow(fitness + 1, 2));
			}
		}

		/// Step 3. advance the epoch
		const bool success = population.epoch_advance();

		statistics.push_back(population.get_epoch_statistics());

		if (!success) {
			print_line("Stopping prematurely: " + brain::itos(epoch));
			break;
		}

		print_line("\nEpoch: " + brain::itos(epoch));
	}

	{ // Print statistics
		std::string s("");
		for (auto it = statistics.begin(); it != statistics.end(); ++it) {
			s += std::string(*it) + ",";
		}
		s.resize(s.size() - 1); // Remove last comma
		s = "[" + s + "]";

		s = "{\"seed\":" + brain::itos(settings.seed) + ", \"statistics\":" + s + "}";

		print_line(s);
	}

	brain::SharpBrainArea ba;
	population.get_champion_network(ba);
	// TODO get the population champion and test it.
	int b = 0;
}

int main() {

	brain::ErrorHandlerList *error_handler = new brain::ErrorHandlerList;
	error_handler->errfunc = print_error_callback;
	brain::add_error_handler(error_handler);

	//test_NEAT_XOR();
	test_uniform_ba_XOR();

	return 0;
}
