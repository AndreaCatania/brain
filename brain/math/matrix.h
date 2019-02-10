#pragma once

#include "brain/math/math_defs.h"
#include "brain/string.h"

typedef real_t (*matrix_map)(real_t p_val);
typedef real_t (*matrix_map_a1)(real_t p_val, real_t p_arg1);

// TODO make this class COW please
namespace brain {
class Matrix {

	uint32_t rows;
	uint32_t columns;
	real_t *matrix;

public:
	Matrix();
	Matrix(
			const uint32_t p_rows,
			const uint32_t p_columns,
			const real_t *const p_matrix = nullptr);

	Matrix(const Matrix &p_other);

	~Matrix();

	void resize(const uint32_t p_rows, const uint32_t p_columns);

	uint32_t get_row_count() const { return rows; }
	uint32_t get_column_count() const { return columns; }

	/**
	 * Set the matrix using an array
	 * The array is split in rows depending on the column count.
	 *
	 * This is marked as unsafe because it accept a pointer and there is no way
	 * to know if the passed array has the correct dimension.
	 * The used must know how to use it.
	 */
	void unsafe_set(const real_t *const p_matrix);
	void unsafe_set_row(const uint32_t p_row, const real_t *const p_data);

	void set(int p_row, int p_column, real_t p_value);
	inline real_t get(int p_row, int p_column) const;

	void set_all(real_t p_value);

	const real_t *get_matrix() const { return matrix; }

	// Map each element in the matrix
	void map(matrix_map p_func);
	Matrix mapped(matrix_map p_func) const;

	void map(matrix_map_a1 p_func, real_t p_arg1);
	Matrix mapped(matrix_map_a1 p_func, real_t p_arg1) const;

	real_t summation() const;

	/**
	 * Perform a element wise multiplication
	 */
	void element_wise_multiplicate(const Matrix &p_other);
	Matrix element_wise_multiplicated(const Matrix &p_other);

	void transpose();
	Matrix transposed() const;

	void operator=(const Matrix &p_other);

	Matrix operator*(const Matrix &p_other) const;

	void operator*=(real_t p_num) const;
	Matrix operator*(real_t p_num) const;

	void operator+=(const Matrix &p_other);
	Matrix operator+(const Matrix &p_other) const;

	void operator-=(const Matrix &p_other);
	Matrix operator-(const Matrix &p_other) const;

	operator std::string() const;

private:
	void init();
	void free();
};
} // namespace brain
