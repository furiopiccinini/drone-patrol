import h5py
import numpy as np
from struct import *


class Metadata():
    def __init__(self, metadata):
        self.input_dimension = metadata["Input dimension"][()]
        self.max_input_values = list(metadata["Maximum input values"])
        self.max_output_values = list(metadata["Maximum output values"])
        self.min_input_values = list(metadata["Minimum input values"])
        self.min_output_values = list(metadata["Minimum output values"])
        self.output_dimension = metadata["Output dimension"][()]
        self.task_type = metadata["Task type"][0].decode("utf-8").upper()
        if (self.task_type == "CLASSIFICATION"):
            self.classes = list(metadata["Classes"])
            self.classes_map = {}
            for i in range(len(self.classes)):
                self.classes_map[self.classes[i]] = i


class CalculateInfo():
    def __init__(self, neuron_data):
        calculate_info = neuron_data["Calculate Info"]
        self.function_coefficents = list(
            calculate_info["Functions coefficients"])
        metadata = neuron_data["Metadata"]

        self.neuron_count = len(self.function_coefficents)
        self.output_labels = list(calculate_info["Output labels"])
        self.weights = self.unpack_weights(calculate_info, metadata)
        self.train_param_count = self.neuron_count
        self.train_param_count += self.weights.size

    def unpack_weights(self, calculate_info, metadata):
        weights_list = list(calculate_info["Weights"])
        positions_packed = list(calculate_info["Positions"])

        rows = len(self.function_coefficents)
        cols = len(self.function_coefficents) + metadata["Input dimension"][()]

        len_w = cols * rows
        positions_type_size = 1
        if len_w > 256:
            if len_w > 65536:
                positions_type_size = 4
            else:
                positions_type_size = 2

        positions_unpacked = []
        if positions_type_size == 4:
            positions_unpacked = positions_packed
        else:
            pack_format = 'BBBB' if positions_type_size == 1 else 'HH'
            for int_packed in positions_packed:
                packed = pack('I', int_packed)
                unpacked = unpack(pack_format, packed)
                for unpacked_el in unpacked:
                    positions_unpacked.append(unpacked_el)

        weights = np.zeros(shape=[rows, cols])
        for id in range(len(weights_list)):
            true_id = positions_unpacked[id]
            row = true_id // cols
            col = true_id % cols
            weights[row, col] = weights_list[id]

        return weights


class NeutonNet():
    def __init__(self, filename):
        neuron_data = dict(h5py.File(filename, 'r'))
        self.calculate_info = CalculateInfo(neuron_data)
        self.metadata = Metadata(neuron_data["Metadata"])

    def predict(self, x: np.ndarray) -> np.ndarray:
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
            sum = [0.] * x.shape[0]
            sum = np.array(sum, dtype=float)

            for ind in range(neuron_index):
                first_elem = job_net_states[:, ind]
                second_elem = self.calculate_info.weights[neuron_index, ind]
                sum = np.sum([sum, first_elem * second_elem], axis=0)

            sum = np.sum([sum, pre_calc[:, neuron_index]], axis=0)
            mulsum = np.greater(abs(-self.calculate_info.function_coefficents[neuron_index] * sum), 700)
            index = np.where(mulsum)
            sum[index] = 700

            if self.calculate_info.function_coefficents[0].size == 1:
                job_net_states[:, neuron_index] = 1.0 / (
                        1.0 + np.exp(-self.calculate_info.function_coefficents[neuron_index][0] * sum))
            elif self.calculate_info.function_coefficents[0].size == 2:
                job_net_states[:, neuron_index] = (1.0 / (
                        1.0 + np.exp(-self.calculate_info.function_coefficents[neuron_index][0] * sum))) - \
                                                  self.calculate_info.function_coefficents[neuron_index][1]
            elif self.calculate_info.function_coefficents[0].size == 3:
                job_net_states[:, neuron_index] = (1.0 / (
                        1.0 + np.exp(-self.calculate_info.function_coefficents[neuron_index][0] * sum +
                                     self.calculate_info.function_coefficents[neuron_index][1]))) - \
                                                  self.calculate_info.function_coefficents[neuron_index][2]

            for ind in range(x.shape[0]):
                job_net_states[ind, neuron_index] = 0.0 if job_net_states[ind, neuron_index] < 0 else (
                    1.0 if job_net_states[ind, neuron_index] > 1.0 else job_net_states[ind, neuron_index])

        calc = list(map(lambda x: x.real, job_net_states[:]))
        predicted = np.asarray(calc)[:, self.calculate_info.output_labels]
        return predicted, job_net_states

    def transform_input(self, input: np.ndarray) -> np.ndarray:
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
        output = output.astype(float)
        for idx in range(self.metadata.output_dimension):
            amp = abs(self.metadata.max_output_values[idx] - self.metadata.min_output_values[idx])
            output[:, idx] = output[:, idx] * amp + self.metadata.min_output_values[idx]
        return output
