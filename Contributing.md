# Extending the project

This file explains how to extend the project. The following cases will be mentioned:

- Adding a page
- Adding a View to an existing page
- Adding a Mining Algorithm
- Add new column selections
- Extending Models

## Adding a page

The project uses the MVC pattern to build pages. At least 2 classes need to be written, a view class and a controller class. These classes should be stored in its own folder inside the ui package. The logic for a page should be store in models. These need to be created, or existing models can be reused. Views are only responsible for displaying the page. The controller decides what part of the view or what view is displayed, processes input events, e.g. button clicks or value changes, calls the models and passes data to the view class to display it. It is recommended to use the template stored in `templates/ui_template` to create a new page.

The `run` method inside the controller is run at every reload, the `process_session_state` method should be used to either read values from the session state or to set values in the session state. `select_view` is an optional method and only needs to be implemented when more than one view is used for a page, the `process_session_state` method is called before the `run` and `select_view` method. Variables set in this method can be directly used in the `select_view` and `run` method.

It is recommended to have a function for each section that should be displayed, inside the view class. The `create_layout` method can be uses to reserve space for a section and render these sections out of order.

To add the page to the application, the controller needs to be called in the `streamlit_app.py` file, with a route.

## Adding a View to an existing page

If a new view for a page is added, the view should be stored in the same folder as the page views and the controller. The view should either be a subclass of the BaseView class or directly inherit from the page view class. The switch between views the `select_view` method has to be implemented. If the two views do not have the same methods, the run methods also need to be changed to support the new view. An example of the `select_view` method is displayed here:

```python
def select_view(self):
        if self.selected_algorithm == "heuristic":
            return self.views[1],1
        else:
            return self.views[0], 0
```

One use case will be adding a ColumnSelectionView. This case will be mentioned here, due to its importance.

## Adding new column selections

Adding new column selections may be necessary in the future, for some algorithms, e.g. a resource column, a person column. To add these columns, there are two templates in the `templates/column_selection_template`. The `BaseColumnSelectionViewTemplate` does not contain any columns at all, and all columns need to be added by the new class. The `ExtendedColumnSelectionViewTemplate` contains the time_column, case_column and event_column/activity_column. The `needed_columns` and `column_styles` need to be added with additional data. The method `render_column_selections` needs to add a selection box with a key equal to the `needed_column` name (e.g. person_column), so that the controller can assign the value to the correct field. Additionally, the `select_view` method of the `ColumnSelectionController` needs to be updated, to switch to this view, if a specific algorithm has been chosen. To ensure the correct transformation for the mining algorithm, the method `transform_df_to_log` need to be overridden, in the AlgorithmController of the new mining algorithms. Otherwise, the additional columns will be ignored, and the default transformation will be performed.

## Adding a Mining Algorithm

The project uses the MVC pattern to build pages. To add a mining algorithm at least 3 classes are needed, an AlgorithmView, an AlgorithmController and a MiningModel. The view(s) and the controller should be in its own folder in the `ui` directory. It is recommended to use the template provided in `templates/algorithm_ui_template`. The MiningModel should be stored in the `mining_algorithms` folder, and the model should inherit from either the `MiningInterface` or the `BaseMining` class. Developers need to choose which class suits their algorithm best. Additionally, it is recommended to create a custom Graph class, to define styles of nodes and edges directly in it and to make the code more readable. The Graph class should be stored in the graphs/visualization directory, and the `BaseGraph` is the parent class. It is also possible to use the `BaseGraph` directly.

To create a view, the class has to inherit from the `BaseAlgorithmView` class. Inside this base class, most of the code is already defined. For the child class, only the sliders need to be defined. This is done in the `render_sidebar` method. The keys of the sliders have to be equal to keys of the session state, which are read and set in the controller. It is recommended to use the same keys in the `sidebar_values` dictionary.

The controller has to inherit from the `BaseAlgorithmController` class. Inside the constructor, the mining model class has to be defined (class not instance!). Furthermore, the following methods need to be overridden: `process_algorithm_parameters`, `perform_mining`, `have_parameters_changed`, `get_sidebar_values`. `process_algorithm_parameters` reads the parameters for the mining algorithm from the session state, or sets them with default values, if they are not set. `perform_mining` calls the model to perform the mining algorithm with the algorithm parameters. `have_parameters_changed` checks if a parameter of the algorithm has changed. If this is not the case, the algorithm does not rerun, as the result stays the same. `get_sidebar_values` is used to set minimum and maximum values for the slider.

All dataframes are transformed to algorithm data, by default a log dictionary. If an algorithm needs another data format or additional data from the dataframe, developers can override the `transform_df_to_log(self, df, **selected_columns)` method. This function takes in the pandas dataframe and the selected columns, where a key is the name of the selected column ('time_column') and the value is the selected column of the dateframe (e.g. 'time'). It is important, that the output values need to be compatible with the constructor of the mining model. The transformation logic should be written in the `DataframeTransformations` class.

The Mining model has to either inherit from the `MiningInterface` or the `BaseMining` class. The models need to have getters for all the parameters, and the method to perform the mining has to store a graph of type `BaseGraph` in the `self.graph` variable.

To add the algorithm to the page, it needs to be added in the `config.py`file. The `algorithm_mappings` map the name of the algorithm to the route, the `algorithm_routes` map the route to the controller class. Both dictionaries need to add the new data, to make the algorithm usable. Optionally, a documentation page can be added to the mining algorithm. This should be written in a markdown file and stored in the `docs/algorithms` directory. To add the documentation, the docs path needs to be added to the `docs_path_mappings` dictionary, the key is the route of the algorithms and the value the path to the file.

## Extending Models

This section will be about how to extend model functionality across the application. This will not go over all the models, but only a few important ones.

### Adding new Import Formats

Adding new import formats need a few changes to work. First, the file type needs to be added to the config. This is done by adding it to the `import_file_types_mapping` in the config file. Secondly, the `HomeController` needs to be updated, so that it will process the file format. The format has to be added in the `process_file` method. The `ImportOperations` needs a method to read the new file type.

### Adding a new Export Format

To add a new export format, the `graph_export_mime_types` variable inside the config file needs to be updated with the new file type and its mime type. The `ExportController` needs to be updated to allow the export of the file type, and the `ExportOperations` class needs a function to export the graph to the disk in the correct format.

### Updating the PredictionModel

The `PredictionModel` uses a dictionary to assign columns to a specific type. This prediction data could be updated by adding new values to the dictionary. Another solution would be to use an AI Model for predictions. If the second approach is uses either update the current `PredictionModel` or create a new class, but the function signature should stay the same, if possible, to integrate the new code easier.
