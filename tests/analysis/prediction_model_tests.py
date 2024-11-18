import unittest
from parameterized import parameterized
from analysis.predictions_model import PredictionModel

column_types_predictions_values = {
    "time": set(["time", "date"]),
    "event": set(["event", "activity", "action", "task", "operation"]),
    "case": set(["case", "process", "instance", "session"]),
}


class TestPredictionModel(unittest.TestCase):
    def setUp(self):
        self.prediction_model = PredictionModel(column_types_predictions_values)

    @parameterized.expand(
        [
            ("time_column", "time"),
            ("date_column", "time"),
            ("event_column", "event"),
            ("activity_column", "event"),
            ("case_column", "case"),
            ("process_column", "case"),
            ("person_column", None),
            ("time_label", "time"),
            ("event_label", "event"),
            ("case_label", "case"),
            ("time_label_column", "time"),
            ("event_label_column", "event"),
            ("case_label_column", "case"),
        ]
    )
    def test_column_type_prediction_returns_correct_type(
        self, column_name, expected_type
    ):
        self.assertEqual(
            self.prediction_model.predict_column_type(column_name), expected_type
        )

    @parameterized.expand(
        [
            (
                "time",
                ["timestamp", "datetime", "event", "case"],
                ["timestamp", "datetime"],
            ),
            (
                "event",
                ["timestamp", "datetime", "event", "case"],
                ["event"],
            ),
            (
                "case",
                ["timestamp", "datetime", "event", "session"],
                ["session"],
            ),
            (
                "person",
                [
                    "timestamp",
                    "event",
                    "case",
                    "person",
                ],
                [],
            ),
        ]
    )
    def test_get_predicted_columns_from_type_returns_correct_column_names(
        self, column_type, column_headers, expected_columns
    ):
        self.assertEqual(
            self.prediction_model.get_predicted_columns_from_type(
                column_headers, column_type
            ),
            expected_columns,
        )

    def test_predict_columns_with_known_predictions(self):
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
        ]

        self.assertEqual(
            self.prediction_model.predict_columns(column_headers, needed_columns),
            ["timestamp", "event", "case"],
        )

    def test_predict_columns_with_unknown_predictions(self):
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
            self.prediction_model.predict_columns(column_headers, needed_columns),
            ["timestamp", "event", "case", None],
        )

    def test_predict_columns_not_contains_duplicates_returns_other_value_of_type(self):
        column_headers = [
            "timestamp",
            "datetime",
            "event",
            "case",
            "person",
        ]
        needed_columns = [
            "time_column",
            "date_column",
            "event_column",
            "case_column",
        ]

        self.assertEqual(
            self.prediction_model.predict_columns(column_headers, needed_columns),
            ["timestamp", "datetime", "event", "case"],
        )

    def test_predict_columns_not_contains_duplicates_returns_none_if_only_one_value_of_type(
        self,
    ):
        column_headers = [
            "timestamp",
            "event",
            "case",
            "person",
        ]
        needed_columns = [
            "time_column",
            "date_column",
            "event_column",
            "case_column",
        ]

        self.assertEqual(
            self.prediction_model.predict_columns(column_headers, needed_columns),
            ["timestamp", None, "event", "case"],
        )
