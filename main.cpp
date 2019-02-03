

#include "brain/areas/brain_area.h"
#include "brain/error_macros.h"
#include "brain/math/dynamic_matrix.h"
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

void test_dynamic_matrix() {
	print_line("Hello brain");

	uint64_t seed = time(nullptr);

	brain::DynamicMatrix weights(2, 2);
	weights.randomize(2, &seed);

	real_t matrix_f[] = {
		5,
		7
	};
	brain::DynamicMatrix features(2, 1, matrix_f);

	real_t matrix_b[] = {
		1,
		1
	};
	brain::DynamicMatrix biases(2, 1, matrix_b);

	brain::DynamicMatrix res = weights * features + biases;
	res.sigmoid();

	print_line("weights");
	print_line(weights);
	print_line("features");
	print_line(features);
	print_line("res");
	print_line(res);
}

void test2_dynamic_matrix() {

	real_t matrix_f[] = {
		1,
		1
	};
	brain::DynamicMatrix features(1, 1, matrix_f);

	brain::DynamicMatrix res;
	res = features;
	res = (res * res) * res;

	print_line("res");
	print_line(res);
}

void test_brain_area_train(brain::BrainArea &area) {
	//area.
}

void test_brain_area() {

	uint64_t seed = time(nullptr);

	// Create brain area
	brain::BrainArea area1;
	area1.set_input_layer_size(4);
	area1.set_output_layer_size(4);
	area1.resize_hidden_layers(1);
	area1.set_hidden_layer_size(0, 4);
	area1.randomize_weights(2.0, seed);
	area1.set_biases(1.0);

	brain::DynamicMatrix features(4, 1, { 1, 0, 0, 0 });

	brain::DynamicMatrix result;
	area1.execute(features, result);

	print_line("Result:");
	print_line(result);
}

int main() {

	brain::ErrorHandlerList *error_handler = new brain::ErrorHandlerList;
	error_handler->errfunc = print_error_callback;
	brain::add_error_handler(error_handler);

	//test_dynamic_matrix();
	//test2_dynamic_matrix();
	test_brain_area();

	return 0;
}
