

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

	area1.set_hidden_layer(0, 2, brain::UniformBrainArea::ACTIVATION_SIGMOID);

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
	settings.genetic_mate_singlepoint_threshold = 0.f;
	//genetic_mutate_add_link_recurrent_prob

	/// Step 1. Population creation
	brain::NtPopulation population(
			brain::NtGenome(3, 1, true),
			150 /*population size*/,
			settings);

	const int epoch_max(100);
	for (int epoch(0); epoch < epoch_max; ++epoch) {
		//for (int epoch(0); true; ++epoch) {

		/// Step 2. Population testing and evaluation
		for (int i = 0; i < population.get_population_size(); ++i) {
			const brain::SharpBrainArea *brain_area = population.organism_get_network(i);
			brain::Matrix result;
			int acceptable_result(0);
			real_t total_error(0);
			for (int k(0); k < inputs.size(); ++k) {
				brain_area->guess(inputs[k], result);
				real_t error = result.get(0, 0) - expected[k].get(0, 0);
				error = ABS(error);
				if (error < 0.3) {
					++acceptable_result;
				}
				total_error += error;
			}

			real_t fitness = 1.f - (total_error / inputs.size());
			population.organism_set_fitness(i, fitness);
			if (fitness > 1.2) { // TODO remove this:
				int a = 0;
			}
			//if (acceptable_result != inputs.size()) {
			//
			//	population.organism_set_fitness(i, fitness);
			//} else {
			//
			//	population.organism_set_fitness(i, brain::Math::pow(fitness, 7));
			//}

			if (total_error <= 0.01) {
				for (int k(0); k < inputs.size(); ++k) {
					brain_area->guess(inputs[k], result);
					real_t error = result.get(0, 0) - expected[k].get(0, 0);
					int a = 0;
				}
				break;
			}
		}

		/// Step 3. advance the epoch
		const bool success = population.epoch_advance();
		if (!success) {
			print_line("Stopping prematurely: " + brain::itos(epoch));
			break;
		}

		print_line("\nEpoch: " + brain::itos(epoch));
		print_line("Pop best fitness: " + brain::rtos(population.get_best_personal_fitness()));
	}

	// TODO get the population champion and test it.
	int b = 0;
}

int main() {

	brain::ErrorHandlerList *error_handler = new brain::ErrorHandlerList;
	error_handler->errfunc = print_error_callback;
	brain::add_error_handler(error_handler);

	/*
	brain::Math::randomize();

	brain::NtGenome genome;
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_INPUT);
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_HIDDEN);
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_HIDDEN);
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_INPUT);
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_HIDDEN);
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_HIDDEN);
	genome.add_neuron(brain::NtNeuronGene::NEURON_GENE_TYPE_OUTPUT);
	genome.add_link(0, 1, brain::Math::randd(), false, 1);
	genome.add_link(1, 2, brain::Math::randd(), false, 2);
	genome.add_link(2, 6, brain::Math::randd(), false, 3);
	genome.add_link(3, 4, brain::Math::randd(), false, 4);
	genome.add_link(4, 5, brain::Math::randd(), false, 5);
	genome.add_link(5, 6, brain::Math::randd(), false, 6);
	genome.add_link(2, 5, brain::Math::randd(), false, 7);
	genome.add_link(5, 1, brain::Math::randd(), true, 8);
	genome.add_link(6, 6, brain::Math::randd(), true, 9);

	brain::NtGenome genome2;
	genome.duplicate_in(genome2);

	std::vector<brain::NtInnovation> innovations;
	uint32_t inn = 9;

	genome2.mutate_add_random_neuron(
			innovations,
			inn);

	genome2.mutate_add_random_neuron(
			innovations,
			inn);

	brain::NtGenome genome3;
	//genome3.mate_multipoint(genome, 0.4, genome2, 2, true);
	genome3.mate_singlepoint(genome, genome2);

	brain::SharpBrainArea a;
	genome3.generate_neural_network(a);

	real_t b[] = { 1, 0 };
	brain::Matrix res;
	a.guess(brain::Matrix(2, 1, b), res);

	int bre_ak = 0;
	*/

	test_NEAT_XOR();
	//test_uniform_ba_XOR();

	return 0;
}
