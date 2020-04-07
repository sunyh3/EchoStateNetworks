import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List


class Data:
    def __init__(
        self,
        model_input: Union[np.ndarray, pd.DataFrame],
        index: list = None,
        name_data_set: str = "Data",
        model_output: Union[np.ndarray, pd.DataFrame] = None,
        x_labels: List[str] = None,
        y_labels: List[str] = None,
    ):

        convert = np.vectorize(lambda x: float(str(x).replace(",", ".")))
        self._time_series_input = convert(model_input)
        self._time_series_output = convert(model_output)
        if model_output is not None:
            assert (
                self._time_series_input.shape[0] == self._time_series_output.shape[0]
            ), "Input and output time series must have same length in dimension 0 "

        self._no_data_points = self._time_series_input.shape[0]
        self._name_data_set = name_data_set
        self._index = index or list(np.arange(0, self._no_data_points))

        self._start_point_index = 0
        self._split_index = self._time_series_input.shape[0]

        self._data_scaled: bool = False
        self._scale = None

        self._x_labels = x_labels or []
        self._y_labels = y_labels or []
        if type(model_input) == pd.DataFrame:
            self._x_labels = model_input

        self._x_train: np.ndarray = np.array([])
        self._x_test: np.ndarray = np.array([])
        self._y_train: np.ndarray = np.array([])
        self._y_test: np.ndarray = np.array([])
        self.__set_data()

    def __set_data(self):
        self._x_train = self._time_series_input[
            self._start_point_index : self._split_index - 1
        ]
        self._x_test = self._time_series_input[self._split_index : -1]

        output_data = (
            self._time_series_output
            if self._time_series_output is not None
            else self._time_series_input
        )
        self._y_train = output_data[self._start_point_index + 1 : self._split_index]
        self._y_test = output_data[self._split_index + 1 :]

    @property
    def no_input_series(self) -> int:
        return self._time_series_input.shape[1]

    @property
    def no_output_series(self) -> int:
        return (
            self._time_series_output.shape[1]
            if self._time_series_output is not None
            else self._time_series_input.shape[1]
        )

    @property
    def no_data_points(self) -> int:
        return self._no_data_points

    @property
    def name(self) -> str:
        return self._name_data_set

    @property
    def index(self) -> list:
        return self._index

    @property
    def x_train(self) -> np.ndarray:
        return self._x_train

    @property
    def y_train(self) -> np.ndarray:
        return self._y_train

    @property
    def x_test(self) -> np.ndarray:
        return self._x_test

    @property
    def y_test(self) -> np.ndarray:
        return self._y_test

    @property
    def scale(self):
        return self._scale

    def visualize(self, data_points: int = 500, show_out_put: bool = False):
        def plot_graph(ts_data):
            if not show_out_put:
                fig, axs = plt.subplots(2, 2)
                k = 0
                for i in [0, 1]:
                    for j in [0, 1]:
                        axs[i, j].plot(ts_data[-data_points:, k])
                        k += 1
                plt.tight_layout()
                plt.show()

        if not show_out_put:
            plot_graph(self._time_series_input)
        else:
            plot_graph(self._time_series_output)

    def split_data_by_index(
        self, split_index: int, start_point_index: int = 0
    ) -> (str, str):
        self._split_index = split_index
        self._start_point_index = start_point_index

        testing_start_label = self._index[self._split_index]
        testing_end_label = self._index[-1]

        self.__set_data()
        return testing_start_label, testing_end_label

    def split_data_by_label(
        self, split_label: str, start_label: str = None
    ) -> (str, str):
        assert split_label in list(self._index), "Label not found"
        self._split_index = list(self._index).index(split_label)
        try:
            self._start_point_index = list(self._index).index(start_label)
        except ValueError:
            self._start_point_index = 0

        testing_start_label = self._index[self._split_index]
        testing_end_label = self._index[-1]
        self.__set_data()
        return testing_start_label, testing_end_label

    def scale_data(self, sk_learn_scaler):
        data_shapes = [
            self._x_train.shape,
            self._y_train.shape,
            self._x_test.shape,
            self._y_test.shape,
        ]

        scaled_training_data = sk_learn_scaler.fit_transform(
            np.append(self._x_train, self._y_train, axis=1)
        )
        scaled_test_data = sk_learn_scaler.transform(
            np.append(self.x_test, self.y_test, axis=1)
        )

        self._x_train = scaled_training_data[:, : self.x_train.shape[1]]
        self._y_train = scaled_training_data[:, -self.y_train.shape[1] :]
        self._x_test = scaled_test_data[:, : self.x_test.shape[1]]
        self._y_test = scaled_test_data[:, -self.y_test.shape[1] :]

        self._scale = sk_learn_scaler
        self._data_scaled = True

        assert data_shapes == [
            self._x_train.shape,
            self._y_train.shape,
            self._x_test.shape,
            self._y_test.shape,
        ], "Data scaling changes data shapes "

    def rescale_data(self):
        assert self._data_scaled == True, "Data is not scaled"
        data_shapes = [
            self._x_train.shape,
            self._y_train.shape,
            self._x_test.shape,
            self._y_test.shape,
        ]

        rescaled_training_data = self._scale.inverse_transform(
            np.append(self._x_train, self._y_train, axis=1)
        )
        rescaled_test_data = self._scale.inverse_transform(
            np.append(self.x_test, self.y_test, axis=1)
        )

        self._x_train = rescaled_training_data[:, : self.x_train.shape[1]]
        self._y_train = rescaled_training_data[:, -self.y_train.shape[1] :]
        self._x_test = rescaled_test_data[:, : self.x_test.shape[1]]
        self._y_test = rescaled_test_data[:, -self.y_test.shape[1] :]

        self._data_scaled = False
        assert data_shapes == [
            self._x_train.shape,
            self._y_train.shape,
            self._x_test.shape,
            self._y_test.shape,
        ], "Data rescaling changes data shapes "
