#pragma once

#include "brain/brain_areas/brain_area.h"
#include "brain/math/matrix.h"
#include <vector>

namespace brain {

/**
 * @brief The UniformBrainArea class is the type of brain area that give the
 * possibility to create fully connected neural network
 *
 * The execution of the guess function is usually two times slower than the
 * sharp brain area, but its creation is faster.
 */
class UniformBrainArea : public brain::BrainArea {

public:
	/**
	 * @brief The DeltaGradients struct holds the delta weight to add for
	 * all the layers.
	 */
	struct DeltaGradients {
		std::vector<Matrix> weights;
		std::vector<Matrix> biases;

		void operator+=(const DeltaGradients &p_other);
		void operator/=(int p_num);
	};

	/**
	 * @brief The LearningCache struct holds the information that are used
	 * during the learning phase
	 */
	struct LearningData {

		/**
		 * @brief layers_input has the not yet actived data
		 */
		std::vector<brain::Matrix> layers_input_signal;

		/**
		 * @brief layers_output has the actived data
		 */
		std::vector<brain::Matrix> layers_output_signal;
	};

private:
	/**
	 * @brief weights, biases, activations, are organized in this way:
	 *
	 * Weights[3]
	 * Biases[3]
	 * Activations[2]
	 *
	 * layer 0            layer 1                         layer 2
	 * | -- weight - bias | -- Activation - weight - bias | -- Activation - weight - bias
	 * | -- weight - bias | -- Activation - weight - bias | -- Activation - weight - bias
	 * | -- weight - bias | -- Activation - weight - bias | -- Activation - weight - bias
	 *
	 * To facilitate the reading use the below macros, they accept the
	 * layer ID and returns the index to use in these arrays.
	 *
	 * INPUT_INDEX
	 * HIDDEN_INDEX(layer)
	 * OUTPUT_INDEX
	 * ACTIVATION_INDEX(layer)
	 * WEIGHT_INDEX(layer)
	 * BIAS_INDEX(layer)
	 */
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;
	std::vector<Activation> activations;

public:
	UniformBrainArea();
	UniformBrainArea(
			uint32_t p_input_layer_size,
			uint32_t p_hidden_layers_count,
			uint32_t p_output_layer_size);

	void set_input_layer_size(uint32_t p_size);
	virtual uint32_t get_input_layer_size() const;

	void set_output_layer_size(uint32_t p_size);
	virtual uint32_t get_output_layer_size() const;

	void set_output_layer_activation(Activation p_activation);
	Activation get_output_layer_activation() const;

	void set_hidden_layers_count(uint32_t p_count);
	uint32_t get_hidden_layers_count() const;

	void set_hidden_layer(
			uint32_t p_hidden_layer,
			uint32_t p_size,
			Activation p_activation);

	void set_hidden_layer_size(uint32_t p_hidden_layer, uint32_t p_size);
	uint32_t get_hidden_layer_size(uint32_t p_hidden_layer) const;

	void set_hidden_layer_activation(uint32_t p_layer, Activation p_activation);
	Activation get_hidden_layer_activation(uint32_t p_layer) const;

	virtual void randomize_weights(real_t p_range);
	virtual void fill_weights(real_t p_value);

	void randomize_biases(real_t p_range);
	void fill_biases(real_t p_value);

	int get_layer_count() const;

	void set_layer_weights(int p_layer, const Matrix &p_matrix);
	const Matrix &get_layer_weights(const int p_layer) const;

	void set_layer_biases(int p_layer, const Matrix &p_matrix);

	void set_layer_activation(int p_layer, Activation p_activation);

	const std::vector<Matrix> &get_weights() const { return weights; }
	const std::vector<Matrix> &get_biases() const { return biases; }
	const std::vector<Activation> &get_activations() const { return activations; }

	/**
	 * @brief learn
	 * @param p_guess
	 * @param p_expected
	 * @param p_learn_rate usually something around 0.05
	 * @param p_update_weights if false the weights will not updated.
	 *			Useful when you need to take the delta weight
	 * @param r_gradients if not null, it's filled with the delta gradients
	 *			calculated, then is possible to use them in
	 * @param r_cache if null the cache is cleared for each call
	 * @return Returns the error of this guess, 0 == Accurate
	 *
	 * This function can be used to train the brain area.
	 * Currently it uses the stochastic gradient descent algorithm
	 * to adjust itw weights
	 */
	real_t learn(
			const Matrix &p_input,
			const Matrix &p_expected,
			real_t p_learn_rate,
			bool p_update_weights = true,
			DeltaGradients *r_gradients = NULL,
			LearningData *r_cache = NULL);

	/**
	 * @brief update_weights can be used to updated the weights using the DeltaGradients.
	 * This function is useful to perform Batch or mini batch gradient descent
	 *
	 * @param p_gradients
	 */
	void update_weights(const DeltaGradients &p_gradients);

	/**
	 * @brief guess
	 * @param p_input Input data
	 * @param r_guess result
	 * @param r_ld is the learning data, pass null if you don't need to know
	 *			these info
	 */
	bool _guess(
			const Matrix &p_input,
			Matrix &r_guess,
			LearningData *r_ld = nullptr) const;

	virtual bool guess(
			const Matrix &p_input,
			Matrix &r_guess) const;

	/**
	 * @brief The MetadataIndices enum
	 * First is an uint32_t with the size of the entire buffer
	 * Second is an uint32_t that point the size of the real_t
	 * Third is an uint32_t with the weight count
	 * Forth is an uint32_t with the biases count
	 * Fifth is an uint32_t with the activation count
	 * From now on all arrays store in this order weights, biases, activations
	 */
	enum MetadataIndices {
		METADATA_BUFFER_SIZE,
		METADATA_REAL_SIZE,
		METADATA_WEIGHT_COUNT,
		METADATA_BIAS_COUNT,
		METADATA_ACTIVATION_COUNT,
		METADATA_MAX
	};

	/**
	 * @brief get_buffer_metadata_size
	 * @param p_buffer_metadata
	 * @return
	 *
	 * Returns the size of the metadata
	 */
	virtual int get_buffer_metadata_size() const;

	/**
	 * @brief get_buffer_size
	 * @param p_buffer_metadata
	 * @return
	 *
	 * Read the metadata and returns the size of the entire buffer
	 */
	virtual uint32_t get_buffer_size(const std::vector<uint8_t> &p_buffer_metadata) const;

	/**
	 * @brief is_buffer_corrupted
	 * @param p_buffer
	 * @return
	 *
	 * Return true if the buffer is corrup
	 */
	virtual bool is_buffer_corrupted(const std::vector<uint8_t> &p_buffer) const;

	/**
	 * @brief is_buffer_compatible
	 * @param p_buffer
	 * @return
	 *
	 * This function returns true if the buffer is compatible with the
	 * current structure
	 */
	virtual bool is_buffer_compatible(const std::vector<uint8_t> &p_buffer) const;

	/**
	 * @brief set_buffer
	 * @param p_buffer
	 * @return
	 *
	 * Restore the weights the biases and activations.
	 *
	 * This function alter the current structure of the brain area,
	 * use the function is_buffer_compatible to know if this buffer
	 * is compatible with the current structure
	 */
	virtual bool set_buffer(const std::vector<uint8_t> &p_buffer);

	/**
	 * @brief get_buffer return a buffer with the current knowledge
	 * @param p_buffer
	 * @return
	 */
	virtual bool get_buffer(std::vector<uint8_t> &r_buffer) const;

private:
	void set_layer_size(uint32_t p_layer, uint32_t p_size);
	uint32_t get_layer_size(uint32_t p_layer) const;
};

} // namespace brain
