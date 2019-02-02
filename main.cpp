

#include "brain/error_macros.h"
#include "brain/math/dynamic_matrix.h"
#include "brain/string.h"
#include "brain/typedefs.h"
#include <time.h>

void print_line(const std::string &p_msg) {
	printf(p_msg.c_str());
	printf("\n");
}

int main() {

	print_line("Hello brain");

	uint64_t seed = time(nullptr);

	brain::DynamicMatrix weights(2, 2);
	weights.randomize(2, &seed);

	real_t matrix_f[] = {
		0.4,
		0.8
	};
	brain::DynamicMatrix features(2, 1, matrix_f);

	real_t matrix_b[] = {
		1,
		1
	};
	brain::DynamicMatrix biases(2, 1, matrix_b);

	brain::DynamicMatrix res = weights * features + biases;
	res.sigmoid();

	print_line(weights);
	print_line(features);
	print_line(res);
	return 0;
}
