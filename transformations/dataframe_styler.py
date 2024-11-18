import pandas as pd


class DataFrameStyler:

    def __init__(self, max_rows_shown: int) -> None:
        """create a new DataframeStyler object

        Parameters
        ----------
        max_rows_shown : int
            the maximum number of rows to show in the styled dataframe
        """
        self.max_rows_shown = max_rows_shown
        self.df = None
        self.column_styles = {}
        self.default_style = ""

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """set the dataframe to style

        Parameters
        ----------
        df : pd.DataFrame
            the dataframe to style
        """
        self.df = df

    def set_column_styles(self, column_styles: dict[str, str]) -> None:
        """set the column styles for the dataframe

        Parameters
        ----------
        column_styles : dict[str, str]
            a dictionary, mapping selected column names to styles
        """
        self.column_styles = column_styles

    def set_default_style(self, default_style: str) -> None:
        """set the default style for the dataframe

        Parameters
        ----------
        default_style : str
            the default style for the dataframe
        """
        self.default_style = default_style

    def stlye_df(
        self,
        selected_columns: dict[str, str],
    ) -> pd.DataFrame:
        """function to style the dataframe.
        It applies the styles to the selected columns.
        The rest of the columns are styled with the default style.

        Parameters
        ----------
        selected_columns : dict[str, str]
            a dictionary, mapping the name of a selection (e.g. 'time_column') to the selected column name

        Returns
        -------
        pd.DataFrame
            the styled dataframe
        """
        reduced_df = self.df.head(self.max_rows_shown)
        styled_df = reduced_df.style.apply(
            axis=0,
            func=self.style_df_columns,
            selected_columns=selected_columns,
        )

        return styled_df

    def style_df_columns(
        self, column: pd.Series, selected_columns: dict[str, str]
    ) -> list[str]:
        """function to style the columns of the dataframe.

        Parameters
        ----------
        column : pd.Series
            column of the dataframe to style
        selected_columns : dict[str, str]
            a dictionary, mapping the name of a selection (e.g. 'time_column') to the selected column name

        Returns
        -------
        list[str]
            a list of styles for the column
        """
        if column.name not in selected_columns.values():
            return [self.default_style] * len(column)

        for needed_column, selected_column in selected_columns.items():
            if column.name == selected_column:
                return [
                    self.column_styles.get(needed_column, self.default_style)
                ] * len(column)
