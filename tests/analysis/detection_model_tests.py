import unittest
from parameterized import parameterized
from analysis.detection_model import DetectionModel
from exceptions.io_exceptions import UnsupportedFileTypeException

file_types_mapping = {
    "csv": [".csv"],
    "pickle": [".pickle", ".pkl"],
}

graph_export_mime_types = {
    "svg": "image/svg",
    "png": "image/png",
    "dot": "text/plain",
}


class TestDetectionModel(unittest.TestCase):
    def setUp(self):
        self.detection_model = DetectionModel(
            file_type_mappings=file_types_mapping,
            mime_type_mappings=graph_export_mime_types,
        )

    @parameterized.expand(
        [
            ("test.csv", "csv"),
            ("dir/test.csv", "csv"),
            ("tests/Callcenter.csv", "csv"),
        ]
    )
    def test_detect_file_type_returns_csv(self, file_path, expected):
        self.assertEqual(self.detection_model.detect_file_type(file_path), expected)

    @parameterized.expand(
        [
            ("test.pickle", "pickle"),
            ("dir/test.pickle", "pickle"),
            ("tests/Callcenter.pickle", "pickle"),
            ("tests/Callcenter.pkl", "pickle"),
        ]
    )
    def test_detect_file_type_returns_pickle(self, file_path, expected):
        self.assertEqual(self.detection_model.detect_file_type(file_path), expected)

    @parameterized.expand(
        [
            ("test.svg",),
            ("test.xslx",),
            ("test.pdf",),
            ("test.txt",),
        ]
    )
    def test_not_supported_file_type_raises_exception(self, file_path):
        with self.assertRaises(UnsupportedFileTypeException):
            self.detection_model.detect_file_type(file_path)

    @parameterized.expand(
        [
            ("A,B,C",),
            ("time, event, case",),
        ]
    )
    def test_colon_delimiter_detection(self, header_row):
        self.assertEqual(self.detection_model.detect_delimiter(header_row), ",")

    @parameterized.expand(
        [
            ("A;B;C", ";"),
            ("time; event; case", ";"),
            ("A B C", " "),
            ("time event case", " "),
            ("A:B:C", ":"),
            ("time:event:case", ":"),
        ]
    )
    def test_delemiter_detection_with_other_delimiters(
        self, header_row, expected_delimiter
    ):
        self.assertEqual(
            self.detection_model.detect_delimiter(header_row), expected_delimiter
        )

    def test_no_detected_delimiter_returns_empty_string(self):
        self.assertEqual(
            self.detection_model.detect_delimiter(
                "A;B;C\n 1;2;3\n1;2;3\n1;2;3\n1;2;3\n1;2;3\n1;2;3\n1;"
            ),
            "",
        )

    @parameterized.expand(
        [
            ("svg", "image/svg"),
            ("png", "image/png"),
            ("dot", "text/plain"),
        ]
    )
    def test_mime_type_detection_returns_correct_type(
        self, file_type, expected_mime_type
    ):
        self.assertEqual(
            self.detection_model.detect_mime_type(file_type), expected_mime_type
        )

    def test_invalide_mime_type_raises_exception(self):
        with self.assertRaises(UnsupportedFileTypeException):
            self.detection_model.detect_mime_type("pdf")

    @parameterized.expand(
        [
            ("test.pdf", "pdf"),
            ("test.csv", "csv"),
            ("test.pickle", "pickle"),
            ("test.pkl", "pickle"),
            ("test.xlsx", "excel"),
            ("test.xls", "excel"),
            ("test.xlsm", "excel"),
            ("test.odt", "excel"),
        ]
    )
    def test_detection_model_with_custom_file_mappings_return_correct_types(
        self, file_path, expected_file_type
    ):
        custom_file_mapping = file_types_mapping | {
            "pdf": [".pdf"],
            "excel": [".xlsx", ".xls", ".xlsm", "odt"],
        }
        detection_model = DetectionModel(
            file_type_mappings=custom_file_mapping,
            mime_type_mappings=graph_export_mime_types,
        )
        self.assertEqual(
            detection_model.detect_file_type(file_path), expected_file_type
        )

    def test_detection_model_with_custom_graph_export_mime_types_return_correct_types(
        self,
    ):
        custom_graph_export_mime_types = graph_export_mime_types | {
            "pdf": "application/pdf",
        }
        detection_model = DetectionModel(
            file_type_mappings=file_types_mapping,
            mime_type_mappings=custom_graph_export_mime_types,
        )
        self.assertEqual(detection_model.detect_mime_type("pdf"), "application/pdf")


if __name__ == "__main__":
    unittest.main()
