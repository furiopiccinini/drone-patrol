#ifndef NEUTON_MODEL_MODEL_H
#define NEUTON_MODEL_MODEL_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Model info */
#define NEUTON_MODEL_HEADER_VERSION 3
#define NEUTON_MODEL_QLEVEL 32
#define NEUTON_MODEL_FLOAT_SUPPORT 1
#define NEUTON_MODEL_TASK_TYPE 0  // multiclass classification
#define NEUTON_MODEL_NEURONS_COUNT 33
#define NEUTON_MODEL_WEIGHTS_COUNT 169
#define NEUTON_MODEL_INPUTS_COUNT 4
#define NEUTON_MODEL_INPUT_LIMITS_COUNT 4
#define NEUTON_MODEL_OUTPUTS_COUNT 9
#define NEUTON_MODEL_LOG_SCALE_OUTPUTS 0
#define NEUTON_MODEL_HAS_CLASSES_RATIO 0
#define NEUTON_MODEL_HAS_NEGPOS_RATIO 0

/* Preprocessing */
#define NEUTON_PREPROCESSING_ENABLED 0
#define NEUTON_DROP_ORIGINAL_FEATURES 0
#define NEUTON_BITMASK_ENABLED 1
#define NEUTON_INPUTS_IS_INTEGER 0
#define NEUTON_MODEL_SA_PRECISION 24

/* Types */
typedef float input_t;
typedef float extracted_feature_t;
typedef float coeff_t;
typedef float weight_t;
typedef double acc_signed_t;
typedef double acc_unsigned_t;
typedef uint8_t sources_size_t;
typedef uint8_t weights_size_t;
typedef uint8_t neurons_size_t;

/* Limits */
static const input_t modelInputMin[] = { 22.059999, 127428, 29, 5305 };
static const input_t modelInputMax[] = { 3.541e+16, 129464, 88, 27543 };

static const uint8_t modelUsedInputsMask[] = { 0x0f };

/* Structure */
static const weight_t modelWeights[] = {
	0.26562503, -0.87820005, 0.40713787, 0.79198152, 0.32243156, -0.1324342,
	-7.1525574e-07, 0.40010059, -0.90580702, -0.9921205, 0.85619098, -1, 0.24672008,
	-0.875, -0.25641692, 1, 1, -0.98790133, 0.28935155, -0.5997014, 1, -1,
	0.94633257, 0.93589717, -0.60749149, 0.83203506, -0.6451664, 0.1963419,
	0.24998474, 0.36080357, 0.29484767, 0.99998665, 0.19931087, -0.77266484,
	-0.70654297, -0.32113358, 0.35035095, -1, 0.87834066, -0.61553955, 0.859375,
	-1, 1, 0, 0, 0, 0, -0.93382418, -0.1549316, -1, 1, 0.8125, 1, -1, 0.34608853,
	-0.010096312, 1, -0.77385771, -0.19200784, 0.5898326, 0, -0.38657871, -0.21184026,
	0.93519771, 0.87543643, -0.55265474, -0.99662638, -1, 0.45350194, 0.97377396,
	-1, -0.75087214, -0.6875, 0.99987793, -0.081016548, 0.69089639, -0.99830902,
	0.25000024, 0.12503541, -1, 0.5, -1, 1, -1, -1, 0.5, -1, -1, -0.0042665303,
	-1, 0.5, 0.26154369, 0.86131775, 0.53668678, -0.25000012, -1, 0.22361656,
	-0.93851137, -1, -1, 0.21582268, -0.85390323, 1, -0.34401023, -1, 0, -0.94628847,
	0.984375, 1, -1, -1, 1, 1, -1, 0.5, 0.625, -0.62499952, -0.875, -1, 1,
	-0.625, 0.875, 2.9802322e-08, 0.25000024, -0.5, -0.77309692, 1, -0.62483382,
	0.84375, 0.36999398, -1, -1, 0, 1, -1, 0.5, 0.34593543, 1, -1, 0.94497794,
	0.8550415, -0.4847095, -0.14306641, -0.9997263, -0.4830189, -1, 0.31423324,
	-1, -1, -1, 1, -0.5, -1, -0.97323072, -0.94862843, 0.99938965, 0.31493497,
	0.88821846, -0.20326588, -0.54477328, -0.18636306, -1, -0.30667114, 0,
	-0.90624404, 1, 1, -1, 0.19970149 };

static const sources_size_t modelLinks[] = {
	0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 0,
	2, 3, 1, 2, 4, 0, 2, 1, 2, 3, 4, 1, 3, 5, 1, 2, 4, 3, 5, 0, 1, 3, 4, 3,
	5, 7, 1, 2, 4, 8, 4, 5, 7, 1, 2, 3, 4, 0, 2, 4, 1, 2, 4, 1, 5, 7, 11, 1,
	4, 3, 4, 5, 10, 3, 4, 0, 5, 13, 4, 4, 14, 4, 5, 1, 4, 5, 16, 4, 5, 6, 10,
	12, 0, 4, 6, 18, 4, 0, 1, 2, 3, 13, 3, 4, 1, 3, 7, 20, 3, 4, 0, 1, 2, 3,
	18, 4, 0, 5, 1, 3, 4, 0, 11, 22, 23, 4, 0, 2, 3, 4, 2, 25, 4, 7, 11, 16,
	20, 0, 4, 7, 10, 27, 4, 13, 16, 21, 1, 4, 3, 13, 20, 29, 4, 0, 1, 2, 3,
	12, 3, 4, 1, 12, 21, 31, 4 };

static const weights_size_t modelIntLinksBoundaries[] = {
	0, 6, 13, 19, 26, 31, 38, 43, 50, 54, 57, 64, 71, 77, 82, 85, 87, 91, 96,
	100, 106, 112, 119, 122, 129, 133, 136, 141, 146, 150, 156, 162, 168 };
static const weights_size_t modelExtLinksBoundaries[] = {
	5, 11, 17, 23, 29, 35, 41, 47, 53, 55, 61, 67, 73, 79, 83, 86, 89, 92,
	98, 101, 108, 114, 120, 125, 130, 134, 137, 143, 147, 152, 157, 164, 169 };

static const coeff_t modelFuncCoeffs[] = {
	30.024952, 39.999985, 39.991623, 40, 40, 39.99971, 40, 39.999992, 40, 40,
	35.012478, 39.999958, 40, 40, 40, 40, 10.075001, 40, 40, 40, 40, 40, 40,
	37.506252, 40, 10.075001, 39.999161, 25.0375, 31.959925, 40, 39.992081,
	39.992676, 40 };
static const uint8_t modelFuncTypes[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0 };

static const neurons_size_t modelOutputNeurons[] = {
	24, 32, 26, 30, 15, 17, 19, 28, 9 };

#ifdef __cplusplus
}
#endif

#endif // NEUTON_MODEL_MODEL_H

