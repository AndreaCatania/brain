#include "matrix.h"

#include "brain/error_macros.h"
#include "brain/math/math_funcs.h"
#include <algorithm>

#define GET_ID(row, col) (row * columns + col)
#define TRANSPOSED_GET_ID(row, col) (col * rows + row)

#define FOREACH                        \
	const int __count_(rows *columns); \
	for (int i(0); i < __count_; ++i)

#define ELEMENT matrix[i]

brain::Matrix::Matrix() :
		rows(0),
		columns(0),
		matrix(nullptr) {}

brain::Matrix::Matrix(
		const uint32_t p_rows,
		const uint32_t p_columns,
		const real_t *const p_matrix) :
		rows(p_rows),
		columns(p_columns),
		matrix(nullptr) {

	init();

	if (nullptr != p_matrix)
		unsafe_set(p_matrix);
}

brain::Matrix::Matrix(const brain::Matrix &p_other) :
		brain::Matrix(p_other.rows, p_other.columns, nullptr) {

	unsafe_set(p_other.matrix);
}

brain::Matrix::~Matrix() {
	free();
}

void brain::Matrix::resize(const uint32_t p_rows, const uint32_t p_columns) {
	free();
	rows = p_rows;
	columns = p_columns;
	init();
}

void brain::Matrix::unsafe_set(const real_t *const p_matrix) {
	if (0 >= rows && 0 >= columns)
		return;
	std::copy(p_matrix, p_matrix + rows * columns, matrix);
}

void brain::Matrix::unsafe_set_row(const uint32_t p_row, const real_t *const p_data) {
	ERR_FAIL_COND(p_row > rows);
	std::copy(p_data, p_data + columns, matrix + p_row);
}

void brain::Matrix::set(int p_row, int p_column, real_t p_value) {
	ERR_FAIL_COND(p_row >= rows);
	ERR_FAIL_COND(p_column >= columns);
	matrix[GET_ID(p_row, p_column)] = p_value;
}

real_t brain::Matrix::get(int p_row, int p_column) const {
	ERR_FAIL_COND_V(p_row >= rows, 0);
	ERR_FAIL_COND_V(p_column >= columns, 0);
	return matrix[GET_ID(p_row, p_column)];
}

void brain::Matrix::set_all(real_t p_value) {
	FOREACH {
		ELEMENT = p_value;
	}
}

void brain::Matrix::map(matrix_map p_func) {
	FOREACH {
		ELEMENT = p_func(ELEMENT);
	}
}

brain::Matrix brain::Matrix::mapped(matrix_map p_func) const {
	Matrix ret(*this);
	ret.map(p_func);
	return ret;
}

void brain::Matrix::map(matrix_map_a1 p_func, real_t p_arg1) {
	FOREACH {
		ELEMENT = p_func(ELEMENT, p_arg1);
	}
}

brain::Matrix brain::Matrix::mapped(matrix_map_a1 p_func, real_t p_arg1) const {
	Matrix ret(*this);
	ret.map(p_func, p_arg1);
	return ret;
}

real_t brain::Matrix::summation() const {
	real_t t(0);
	FOREACH {
		t += ELEMENT;
	}
	return t;
}

void brain::Matrix::element_wise_multiplicate(const Matrix &p_other) {
	ERR_FAIL_COND(get_row_count() != p_other.get_row_count());
	ERR_FAIL_COND(get_column_count() != p_other.get_column_count());

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {

			matrix[GET_ID(r, c)] *= p_other.matrix[GET_ID(r, c)];
		}
	}
}

brain::Matrix brain::Matrix::element_wise_multiplicated(const Matrix &p_other) {
	Matrix res(*this);
	res.element_wise_multiplicate(p_other);
	return res;
}

void brain::Matrix::transpose() {

	if (0 >= rows || 0 >= columns)
		return;

	real_t *new_matrix = new real_t[rows * columns];

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {
			new_matrix[TRANSPOSED_GET_ID(r, c)] = matrix[GET_ID(r, c)];
		}
	}

	const uint32_t trasposed_rows(columns);
	const uint32_t trasposed_columns(rows);

	free();

	rows = trasposed_rows;
	columns = trasposed_columns;
	matrix = new_matrix;
}

brain::Matrix brain::Matrix::transposed() const {
	brain::Matrix ret(*this);
	ret.transpose();
	return ret;
}

void brain::Matrix::operator=(const Matrix &p_other) {
	resize(p_other.rows, p_other.columns);
	unsafe_set(p_other.matrix);
}

brain::Matrix brain::Matrix::operator*(const brain::Matrix &p_other) const {

	brain::Matrix res(get_row_count(), p_other.get_column_count());

	ERR_FAIL_COND_V(get_column_count() != p_other.get_row_count(), res);

	for (int o_c(0); o_c < p_other.get_column_count(); ++o_c) {

		for (int r(0); r < rows; ++r) {
			real_t e(0);
			for (int c(0); c < columns; ++c) {
				e += matrix[GET_ID(r, c)] * p_other.get(c, o_c);
			}
			res.set(r, o_c, e);
		}
	}

	return res;
}

void brain::Matrix::operator*=(real_t p_num) const {
	FOREACH {
		ELEMENT *= p_num;
	}
}

brain::Matrix brain::Matrix::operator*(real_t p_num) const {
	brain::Matrix ret(*this);
	ret *= p_num;
	return ret;
}

void brain::Matrix::operator+=(const brain::Matrix &p_other) {

	ERR_FAIL_COND(get_row_count() != p_other.get_row_count());
	ERR_FAIL_COND(get_column_count() != p_other.get_column_count());

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {

			matrix[GET_ID(r, c)] = matrix[GET_ID(r, c)] + p_other.matrix[GET_ID(r, c)];
		}
	}
}

brain::Matrix brain::Matrix::operator+(const brain::Matrix &p_other) const {
	brain::Matrix ret(*this);
	ret += p_other;
	return ret;
}

void brain::Matrix::operator-=(const brain::Matrix &p_other) {

	ERR_FAIL_COND(get_row_count() != p_other.get_row_count());
	ERR_FAIL_COND(get_column_count() != p_other.get_column_count());

	for (int r(0); r < rows; ++r) {
		for (int c(0); c < columns; ++c) {

			matrix[GET_ID(r, c)] = matrix[GET_ID(r, c)] - p_other.matrix[GET_ID(r, c)];
		}
	}
}

brain::Matrix brain::Matrix::operator-(const brain::Matrix &p_other) const {

	brain::Matrix ret(*this);
	ret -= p_other;
	return ret;
}

brain::Matrix::operator std::string() const {
	std::string s("Matrix:\n");
	for (int r(0); r < rows; ++r) {

		s += "|";
		for (int c(0); c < columns; ++c) {
			if (c != 0)
				s += "   ";
			const real_t v(matrix[GET_ID(r, c)]);
			if (0 <= v)
				s += " ";
			s += brain::rtos(v, 3);
		}
		s += " |\n";
	}
	return s;
}

void brain::Matrix::init() {
	if (0 >= rows || 0 >= columns)
		return;
	matrix = new real_t[rows * columns];
}

void brain::Matrix::free() {
	delete[] matrix;

	rows = 0;
	columns = 0;
	matrix = nullptr;
}
