#pragma once

#include "brain/math/math_defs.h"
#include "brain/string.h"

namespace brain {
class DynamicMatrix {

	uint32_t rows;
	uint32_t columns;
	real_t **matrix;

public:
	DynamicMatrix();
	DynamicMatrix(
			const uint32_t p_rows,
			const uint32_t p_columns,
			const real_t *const p_matrix = nullptr);

	DynamicMatrix(const DynamicMatrix &p_other);

	~DynamicMatrix();

	void set_size(const uint32_t p_rows, const uint32_t p_columns);
	uint32_t get_rows() const { return rows; }
	uint32_t get_columns() const { return columns; }

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

	real_t get(int p_row, int p_column) const;

	void sigmoid();
	DynamicMatrix sigmoided() const;

	void randomize(real_t p_scale = 0.01, uint64_t *p_seed = nullptr);
	DynamicMatrix randomized(real_t p_scale = 0.01, uint64_t *p_seed = nullptr) const;

	DynamicMatrix operator*(const DynamicMatrix &p_other) const;
	void operator+=(const DynamicMatrix &p_other);
	DynamicMatrix operator+(const DynamicMatrix &p_other) const;
	void operator-=(const DynamicMatrix &p_other);
	DynamicMatrix operator-(const DynamicMatrix &p_other) const;

	operator std::string() const;

private:
	void init();
	void free();
};
} // namespace brain
