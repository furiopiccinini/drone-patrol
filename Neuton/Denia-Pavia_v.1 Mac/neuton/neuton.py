import h5py
import numpy as np
from struct import *
import sys
import pandas as pd
import pickle
import os
import timeit

sys.path.append('dict/static_data')
from neuton_data_bin import NeutonNet as NeutonNetDataBin
import math

CI_OPERANDS_PATH = 'dict/confidence_interval_operands.data'
CRED_IND_OPERANDS_PATH = 'dict/credibility_indicator_operands.data'
DATA_BIN_PATH = 'dict/static_data/data.bin'
DATA2_BIN_PATH = 'dict/static_data/data2.bin'


class Metadata():
    def __init__(self, metadata):
        self.input_dimension = metadata["Input dimension"][()]
        self.max_input_values = list(metadata["Maximum input values"])
        self.max_output_values = list(metadata["Maximum output values"])
        self.min_input_values = list(metadata["Minimum input values"])
        self.min_output_values = list(metadata["Minimum output values"])
        self.output_dimension = metadata["Output dimension"][()]
        self.task_type = metadata["Task type"][()].decode("utf-8").upper()
        if ('REGRESSION' in self.task_type):
            self.to_exp_output = True if metadata["Log scale flags"][0] == 1 else False
        if ('CLASSIFICATION' in self.task_type):
            self.classes = [x.decode("utf-8") for x in list(metadata["Classes"])]
            if '1,000000' in self.classes or '1.000000' in self.classes:
                self.classes = [cls[:-7] for cls in self.classes]


class CalculateInfo():
    def __init__(self, weights_filename, meta_data):
        self.metadata = meta_data["Metadata"]

        self.read_weights_file(weights_filename)
        self.neuron_count = len(self.function_coefficents)
        self.weights = self.unpack_weights()

        self.train_param_count = self.neuron_count
        self.train_param_count += self.weights.size

    def read_weights_file(self, weights_filename):
        f = open(weights_filename, 'r')
        a = np.fromfile(f, dtype=np.uint32)

        weights_size = a[0] // 4
        positions_size = a[1] // 4
        coefficients_size = a[2] // 4
        output_labels_size = a[3] // 4

        index_start = 4
        index_end = index_start + weights_size
        self.weights_packed = a[index_start:index_end]
        weights_float_list = []
        for weight in self.weights_packed:
            weights_float_list.append(unpack('f', weight))
        self.weights_packed = np.asarray(weights_float_list)

        index_start = index_end
        index_end = index_start + positions_size
        self.positions_packed = a[index_start:index_end]

        index_start = index_end
        index_end = index_start + coefficients_size
        self.function_coefficents = a[index_start:index_end]
        function_coefficents_float_list = []
        for f_coeff in self.function_coefficents:
            function_coefficents_float_list.append(unpack('f', f_coeff))
        self.function_coefficents = np.asarray(function_coefficents_float_list)

        index_start = index_end
        index_end = index_start + output_labels_size
        self.output_labels = a[index_start:index_end].astype(int)

    def unpack_weights(self):
        rows = len(self.function_coefficents)
        cols = len(self.function_coefficents) + self.metadata["Input dimension"][()]

        len_w = cols * rows
        positions_type_size = 1
        if len_w > 256:
            if len_w > 65536:
                positions_type_size = 4
            else:
                positions_type_size = 2

        positions_unpacked = []
        if positions_type_size == 4:
            positions_unpacked = self.positions_packed
        else:
            pack_format = 'BBBB' if positions_type_size == 1 else 'HH'
            for int_packed in self.positions_packed:
                packed = pack('I', int_packed)
                unpacked = unpack(pack_format, packed)
                for unpacked_el in unpacked:
                    positions_unpacked.append(unpacked_el)

        weights = np.zeros(shape=[rows, cols])
        for id in range(len(self.weights_packed)):
            true_id = positions_unpacked[id]
            row = true_id // cols
            col = true_id % cols
            weights[row, col] = self.weights_packed[id]

        return weights


class NeutonNet:
    def __init__(self, meta_filename, weights_filename):
        meta_data = dict(h5py.File(meta_filename, 'r'))
        self.calculate_info = CalculateInfo(weights_filename, meta_data)
        self.metadata = Metadata(meta_data["Metadata"])

    def __predict_batch(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x)
        x = np.hstack((x, np.full((x.shape[0], 1), 1)))
        job_net_states = [[0.] * self.calculate_info.neuron_count] * x.shape[0]
        pre_calc = [[0.] * self.calculate_info.neuron_count] * x.shape[0]

        job_net_states = np.array(job_net_states, dtype=complex)
        pre_calc = np.array(pre_calc, dtype=float)

        for i in range(self.calculate_info.neuron_count):
            for j in range(self.metadata.input_dimension):
                np_arg = x[:, j] * self.calculate_info.weights[i, self.calculate_info.neuron_count + j]
                pre_calc[:, i] = np.sum([pre_calc[:, i], np_arg], axis=0)

        for neuron_index in range(self.calculate_info.neuron_count):
            sum = np.transpose(np.multiply(job_net_states[:, :neuron_index + 1],
                                           self.calculate_info.weights[neuron_index, :neuron_index + 1])).sum(axis=0)

            sum = np.sum([sum, pre_calc[:, neuron_index]], axis=0)
            mulsum = np.greater(abs(-self.calculate_info.function_coefficents[neuron_index] * sum), 700)
            index = np.where(mulsum)
            sum[index] = 700
            job_net_states[:, neuron_index] = 1.0 / (
                1.0 + np.exp(-self.calculate_info.function_coefficents[neuron_index] * sum))

        calc = list(map(lambda x: x.real, job_net_states[:]))
        predicted = np.asarray(calc)[:, self.calculate_info.output_labels]

        if predicted.shape[1] > 1 and 'CLASSIFICATION' in self.metadata.task_type:
            denominator = np.sum(predicted, axis=1)
            for i in range(predicted.shape[1]):
                predicted[:, i] = predicted[:, i] / denominator

        return predicted

    def __predict(self, x: np.ndarray) -> np.ndarray:
        batch_size = 10000

        if x.shape[0] > batch_size:
            max_ind = x.shape[0] // batch_size if x.shape[0] % batch_size == 0 else x.shape[0] // batch_size + 1
            y_probabilities_all = None

            for i in range(max_ind):
                index_first = i * batch_size

                if x.shape[0] % batch_size == 0 or i != max_ind - 1:
                    index_last = index_first + batch_size
                else:
                    index_last = index_first + x.shape[0] % batch_size

                y_probability = self.__predict_batch(x[index_first:index_last, :])

                if y_probabilities_all is not None:
                    y_probabilities_all = np.concatenate((y_probabilities_all, y_probability), axis=0)
                else:
                    y_probabilities_all = y_probability

                print("{0} batch out of {1} predicted".format(i + 1, max_ind))
            return y_probabilities_all
        else:
            return self.__predict_batch(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.__predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if 'REGRESSION' in self.metadata.task_type:
            if self.metadata.to_exp_output:
                return np.exp(self.transform_output(self.__predict(x)).flatten())
            else:
                return self.transform_output(self.__predict(x)).flatten()
        else:
            y_probability = self.predict_proba(x)
            return self.transform_output_to_labels(np.argmax(y_probability, axis=1))

    def transform_input(self, input: np.ndarray) -> np.ndarray:
        input = np.nan_to_num(input)
        input = input.astype(float)
        for idx in range(self.metadata.input_dimension - 1):  # because of additional ones in input we subtract 1
            input[:, idx] = np.where(input[:, idx] < self.metadata.min_input_values[idx],
                                     self.metadata.min_input_values[idx], input[:, idx])
            input[:, idx] = np.where(input[:, idx] > self.metadata.max_input_values[idx],
                                     self.metadata.max_input_values[idx], input[:, idx])
            amp = abs(self.metadata.max_input_values[idx] - self.metadata.min_input_values[idx])
            input[:, idx] = (input[:, idx] - self.metadata.min_input_values[idx]) / amp
        return input

    def transform_output(self, output: np.ndarray) -> np.ndarray:
        output = np.nan_to_num(output)
        output = output.astype(float)
        for idx in range(self.metadata.output_dimension):
            amp = abs(self.metadata.max_output_values[idx] - self.metadata.min_output_values[idx])
            output[:, idx] = output[:, idx] * amp + self.metadata.min_output_values[idx]
        return output

    def transform_output_to_labels(self, output: np.ndarray) -> np.ndarray:
        label_outputs = np.array([self.metadata.classes[x] for x in output])
        return label_outputs

    def get_classes_labels(self):
        return self.metadata.classes


def get_t_criteria(num_samples, cl=0.95):
    model = NeutonNetDataBin(DATA_BIN_PATH)
    t_args = np.asarray([[num_samples, 1 - cl]])
    t_args = model.transform_input(t_args)
    t = model.predict(t_args)[0]
    t = model.transform_output(t)[0][0]
    return t


def get_confidence_level(mean_y_pred, std_dev, k=0.2):
    z = k * mean_y_pred / std_dev
    if z <= 1:
        return 0.68
    elif z >= 2:
        return 0.95
    else:
        model = NeutonNetDataBin(DATA2_BIN_PATH)
        cl_args = np.asarray([[z]])
        cl_args = model.transform_input(cl_args)
        cl = model.predict(cl_args)[0]
        cl = model.transform_output(cl)
        cl = cl[0][0] * 2
        return cl


def get_confidence_interval(x: np.ndarray, y: np.ndarray) -> list:
    with open(CI_OPERANDS_PATH, 'r') as ci_op_file:
        num_samples_train = int(ci_op_file.readline())
        mean_y_pred = float(ci_op_file.readline())
        term1 = float(ci_op_file.readline())
        sum_squared = [float(s) for s in ci_op_file.readline().split(',')]
        x_means = [float(s) for s in ci_op_file.readline().split(',')]
    cl = get_confidence_level(mean_y_pred, term1)
    t_criteria = get_t_criteria(num_samples_train, cl)

    ci = []
    points_num = x.shape[0]
    for i in range(points_num):
        point = x[i]
        nominator = np.power(np.subtract(point, x_means), 2)
        term2 = np.sum(np.divide(nominator, sum_squared))
        sigma = term1 * math.sqrt(1 + 1 / num_samples_train + term2)
        mul_scaler = y[i] / mean_y_pred if mean_y_pred != 0 else 1
        result = t_criteria * sigma * mul_scaler
        ci.append(result)

    return ci, cl


def get_int_arr_from_str(str_var):
    return np.array([int(c) for c in str_var.strip()])


def bcd_digits(byte):
    return byte >> 4, byte & 0xF


def get_cl_dims_and_codes_bin(config_path):
    with open(config_path, 'r') as f:
        a = np.fromfile(f, dtype=np.uint8)
    prefix = chr(a[0]) + chr(a[1])
    if prefix != 'nb':
        raise ValueError(f'File format error - {config_path}')

    service_id = a[2]
    version_id = a[3]
    byte_order_marker = hex(unpack('H', a[4:6])[0])

    col_num = unpack('I', a[6:10])[0]
    a = a[10:]

    max_vals = np.zeros([col_num])
    for col_ind in range(col_num):
        cur_ind = col_ind * 4
        max_vals[col_ind] = unpack('f', a[cur_ind:cur_ind + 4])[0]
    a = a[col_num * 4:]

    min_vals = np.zeros([col_num])
    for col_ind in range(col_num):
        cur_ind = col_ind * 4
        min_vals[col_ind] = unpack('f', a[cur_ind:cur_ind + 4])[0]
    a = a[col_num * 4:]

    used_col_num = unpack('I', a[:4])[0]
    a = a[4:]

    rows_num = unpack('I', a[:4])[0]
    a = a[4:]

    used_col_index = np.zeros([used_col_num],dtype=int)
    for col_ind in range(used_col_num):
        cur_ind = col_ind * 4
        used_col_index[col_ind] = unpack('I', a[cur_ind:cur_ind + 4])[0]
    a = a[used_col_num * 4:]

    overal_cl_dim = a[0]
    a = a[1:]

    # temporary skip tree index for C++
    a = a[rows_num * 16:]

    cl_codes = np.zeros([rows_num, used_col_num], dtype=int)
    for row in range(rows_num):
        for col_ind in range(used_col_num):
            cl_codes[row, col_ind] = a[row*used_col_num + col_ind]

    return overal_cl_dim, cl_codes, used_col_index


def get_test_codes(cl_dim, x_test):
    def f(x, dim):
        return dim - 1 if np.floor(x * (dim - 1) + 0.5) >= dim else np.floor(x * (dim - 1) + 0.5)

    f_v = np.vectorize(f)
    x_test_codes = x_test.copy()
    x_test_codes = f_v(x_test_codes, cl_dim)
    return x_test_codes.astype(int)


def get_nearest_test_cl_code(cl_dim, cl_codes, test_code):
    cl_idx = np.argmin(np.sum(np.divide(np.absolute(np.subtract(cl_codes, test_code)), cl_dim), axis=1))
    return cl_codes[cl_idx]


def get_centre_by_cl_code(cl_dims, cl_code):
    term1 = np.divide(np.asarray([1] * cl_code.shape[0]), cl_dims - 1)
    centre = np.multiply(term1, cl_code)
    return centre


def calc_feature_influence(model, x_test_row, is_fe, df_columns):
    import io
    from MOD_add_feats_test import add_feats_test
    fi = np.zeros([len(x_test_row), model.metadata.output_dimension])
    x_test_row = np.expand_dims(x_test_row, axis=0)
    x_base = x_test_row
    if is_fe:
        text_trap = io.StringIO()
        sys.stdout = text_trap
        x_base_fe = add_feats_test(pd.DataFrame(x_base, columns=df_columns)).values
        sys.stdout = sys.__stdout__
        y_base = model.predict_proba(x_base_fe)
    else:
        y_base = model.predict_proba(x_base)
    sum_above_zero = 0
    sum_under_zero = 0
    for i in range(x_base.shape[1]):
        x_incremented = x_base.copy()
        increment = 0.01
        if x_incremented[0, i] + increment > 1.0:
            increment = -increment
        x_incremented[0, i] += increment
        if is_fe:
            text_trap = io.StringIO()
            sys.stdout = text_trap
            x_incremented_fe = add_feats_test(pd.DataFrame(x_incremented, columns=df_columns)).values
            sys.stdout = sys.__stdout__
            y_incremented = model.predict_proba(x_incremented_fe)
        else:
            y_incremented = model.predict_proba(x_incremented)
        for j in range(y_base.shape[1]):
            fi[i, j] = (y_incremented[0, j] - y_base[0, j]) / increment
            if fi[i, j] > 0:
                sum_above_zero += fi[i, j]
            else:
                sum_under_zero += -fi[i, j]
    sum_div = sum_above_zero if sum_above_zero > sum_under_zero else sum_under_zero
    for i in range(x_base.shape[1]):
        for j in range(y_base.shape[1]):
            fi[i, j] /= sum_div
    return fi


def get_credibility_indicator(model, x, is_fe, df_columns):
    cl_dim, cl_codes, used_col_index = get_cl_dims_and_codes_bin(CRED_IND_OPERANDS_PATH)
    x_used = x.copy()
    x_used = x_used[:, used_col_index]
    test_codes = get_test_codes(cl_dim, x_used)
    cred_inds = np.zeros([test_codes.shape[0], model.metadata.output_dimension])
    for i in range(test_codes.shape[0]):
        test_code = test_codes[i]
        cl_near_code = get_nearest_test_cl_code(cl_dim, cl_codes, test_code)
        cl_centre_coord = get_centre_by_cl_code(cl_dim, cl_near_code)
        fi = calc_feature_influence(model, x[i], is_fe, df_columns)
        fi = fi[used_col_index]
        coord_diff = np.subtract(x_used[i], cl_centre_coord)
        for out_idx in range(fi.shape[1]):
            z = np.abs(np.sum(np.multiply(fi[:, out_idx], coord_diff)))
            if z >= 1:
                cred_inds[i, out_idx] = 0
            else:
                cred_inds[i, out_idx] = (1 - z / 2) * 100
            if (cl_near_code == test_code).all():
                cred_inds[i, out_idx] = 100
    return np.mean(cred_inds, axis=1)
