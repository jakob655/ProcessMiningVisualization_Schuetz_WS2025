class PredictionModel:
    # this dictionary contains the possible values for each column type. This is used to predict the columns

    def __init__(self, column_types_predictions_values=None):
        """Constructor for the PredictionModel class.

        Parameters
        ----------
        column_types_predictions_values : dict[str, set[str]]
            dictionary that contains the possible values for each column type. This is used to predict the columns.
            if the parameter is None, the default values from the config file will be used.
        """
        if column_types_predictions_values is None:
            from config import (
                column_types_predictions_values as column_types_predictions_values_config,
            )

            column_types_predictions_values = column_types_predictions_values_config
        self.column_types_predictions_values = column_types_predictions_values

    def get_predicted_columns_from_type(
        self, dataframe_columns: list[str], column_type: str
    ) -> list[str]:
        """get all the columns from the dataframe that have a certain type. eg. time, event, case. This is used to predict the columns.
        The method searches for the values from the column_types_predictions_values dictionary in the column names.

        Parameters
        ----------
        dataframe_columns : list[str]
            column names from the dataframe
        column_type : str
            the type of the column that we are looking for. At the moment we have 3 types: time, event, case

        Returns
        -------
        list[str]
            list of columns that have the specified type. eg. all columns that have "time" or "date" in their name.
        """
        if column_type not in self.column_types_predictions_values.keys():
            return []

        predicted_columns = []

        for column in dataframe_columns:
            for value in self.column_types_predictions_values[column_type]:
                if value in column.lower():
                    predicted_columns.append(column)
                    break

        return predicted_columns

    def predict_column_type(self, needed_column_name: str) -> str | None:
        """predict the type of a column based on its name. The method searches for the values from the column_types_predictions_values dictionary in the column name.

        Parameters
        ----------
        needed_column_name : str
            the name of the column that we want to predict the type for. e.g. "time_column"

        Returns
        -------
        str | None
            the type of the column. At the moment we have 3 types: time, event, case. If the type cannot be predicted, it will return None.
        """
        for column_type, values in self.column_types_predictions_values.items():
            for value in values:
                if value in needed_column_name.lower():
                    return column_type

        return None

    def predict_columns(
        self, dataframe_columns: list[str], needed_columns: list[str]
    ) -> list[str | None]:
        """predict the columns based on their names. The method searches for the values from the column_types_predictions_values dictionary in the column names.
        first it predicts the type of the column, then it searches for the columns that have that type.

        Parameters
        ----------
        dataframe_columns : list[str]
            column names from the dataframe
        needed_columns : list[str]
            the name of the selected columns that we want to predict. eg. ["time_column", "event_column", "case_column"]

        Returns
        -------
        list[str | None]
            list of predicted columns. If one of the needed columns cannot be predicted, it will be None.
        """
        predicted_columns = [None for _ in needed_columns]

        for index, needed_column in enumerate(needed_columns):
            column_type = self.predict_column_type(needed_column)
            if column_type is not None:

                predicted_columns_from_type = self.get_predicted_columns_from_type(
                    dataframe_columns, column_type
                )

                if len(predicted_columns_from_type) > 0:
                    for predicted_column in predicted_columns_from_type:
                        if predicted_column not in predicted_columns:
                            predicted_columns[index] = predicted_column
                            break

        return predicted_columns

    def test_prediction_model_with_custom_column_type_predictions(self):
        column_types_predictions_values = {
            "time": {"time", "date"},
            "event": {"event"},
            "case": {"case"},
            "person": {"person"},
        }
        prediction_model = PredictionModel(column_types_predictions_values)

        column_headers = [
            "timestamp",
            "datetime",
            "event",
            "case",
            "person",
        ]
        needed_columns = [
            "time_column",
            "event_column",
            "case_column",
            "person_column",
        ]

        self.assertEqual(
            prediction_model.predict_columns(column_headers, needed_columns),
            ["timestamp", "event", "case", "person"],
        )
