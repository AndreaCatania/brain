#include "dynamic_matrix.h"

#include "core/error_macros.h"
#include "core/math/math_funcs.h"

#define FOREACH                   \
	for (int r(0); r < rows; ++r) \
		for (int c(0); c < columns; ++c)

#define ELEMENT matrix[r][c]

brain::DynamicMatrix::DynamicMatrix() :
		rows(0),
		columns(0),
		matrix(nullptr) {}

brain::DynamicMatrix::DynamicMatrix(
		const uint32_t p_rows,
		const uint32_t p_columns,
		const real_t *const p_matrix) :
		rows(p_rows),
		columns(p_columns),
		matrix(nullptr) {

	ERR_FAIL_COND(0 >= p_rows);
	ERR_FAIL_COND(0 >= p_columns);

	init();

	if (nullptr != p_matrix)
		unsafe_set(p_matrix);
}

brain::DynamicMatrix::DynamicMatrix(const brain::DynamicMatrix &p_other) :
		brain::DynamicMatrix(p_other.rows, p_other.columns, nullptr) {

	ERR_FAIL_COND(0 >= rows);
	ERR_FAIL_COND(0 >= columns);

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {
			matrix[r][c] = p_other.matrix[r][c];
		}
	}
}

brain::DynamicMatrix::~DynamicMatrix() {
	free();
}

void brain::DynamicMatrix::unsafe_set(const real_t *const p_matrix) {
	for (int r(0); r < rows; ++r) {
		unsafe_set_row(r, (p_matrix + r * columns));
	}
}

void brain::DynamicMatrix::unsafe_set_row(const uint32_t p_row, const real_t *const p_data) {
	ERR_FAIL_COND(p_row > rows);

	for (int c(0); c < columns; ++c) {
		matrix[p_row][c] = p_data[c];
	}
}

real_t brain::DynamicMatrix::get(int p_row, int p_column) const {
	ERR_FAIL_COND_V(p_row >= rows, 0);
	ERR_FAIL_COND_V(p_column >= columns, 0);
	return matrix[p_row][p_column];
}

void brain::DynamicMatrix::sigmoid() {
	FOREACH {
		ELEMENT = brain::Math::sigmoid(ELEMENT);
	}
}

brain::DynamicMatrix brain::DynamicMatrix::sigmoided() const {
	brain::DynamicMatrix ret(*this);
	ret.sigmoid();
	return ret;
}

void brain::DynamicMatrix::randomize(real_t p_range, uint64_t *p_seed) {
	if (p_seed) {
		FOREACH {
			ELEMENT = brain::Math::rand_from_seed(p_range, p_seed);
		}
	} else {
		FOREACH {
			ELEMENT = brain::Math::random(-p_range, p_range);
		}
	}
}

brain::DynamicMatrix brain::DynamicMatrix::randomized(real_t p_scale, uint64_t *p_seed) const {
	brain::DynamicMatrix ret(*this);
	ret.randomize(p_scale, p_seed);
	return ret;
}

brain::DynamicMatrix brain::DynamicMatrix::operator*(const brain::DynamicMatrix &p_other) const {

	brain::DynamicMatrix res(get_rows(), p_other.get_columns());

	ERR_FAIL_COND_V(get_columns() != p_other.get_rows(), res);

	for (int o_c(0); o_c < p_other.get_columns(); ++o_c) {

		for (int r(0); r < rows; ++r) {
			real_t e(0);
			for (int c(0); c < columns; ++c) {

				e += matrix[r][c] * p_other.matrix[c][o_c];
			}
			res.matrix[r][o_c] = e;
		}
	}

	return res;
}

void brain::DynamicMatrix::operator+=(const brain::DynamicMatrix &p_other) {

	ERR_FAIL_COND(get_rows() != p_other.get_rows());
	ERR_FAIL_COND(get_columns() != p_other.get_columns());

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {

			matrix[r][c] = matrix[r][c] + p_other.matrix[r][c];
		}
	}
}

brain::DynamicMatrix brain::DynamicMatrix::operator+(const brain::DynamicMatrix &p_other) const {
	brain::DynamicMatrix ret(*this);
	ret += p_other;
	return ret;
}

void brain::DynamicMatrix::operator-=(const brain::DynamicMatrix &p_other) {

	ERR_FAIL_COND(get_rows() != p_other.get_rows());
	ERR_FAIL_COND(get_columns() != p_other.get_columns());

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {

			matrix[r][c] = matrix[r][c] - p_other.matrix[r][c];
		}
	}
}

brain::DynamicMatrix brain::DynamicMatrix::operator-(const brain::DynamicMatrix &p_other) const {

	brain::DynamicMatrix ret(*this);
	ret -= p_other;
	return ret;
}

brain::DynamicMatrix::operator std::string() const {
	std::string s;
	for (int r(0); r < rows; ++r) {

		s += "[";
		for (int c(0); c < columns; ++c) {
			if (c != 0)
				s += ", ";
			s += brain::rtos(matrix[r][c], 3);
		}
		s += "]";
	}
	return s;
}

void brain::DynamicMatrix::init() {
	matrix = new real_t *[rows];
	for (int i(0); i < rows; ++i) {
		matrix[i] = new real_t[columns];
	}
}

void brain::DynamicMatrix::free() {
	for (int i(0); i < rows; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;

	rows = 0;
	columns = 0;
	matrix = nullptr;
}
