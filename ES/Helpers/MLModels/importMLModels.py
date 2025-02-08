from pypmml import Model
import os


# Load the PMML file
current_path = os.path.dirname(os.path.abspath(__file__))

example_ML_model_file_path = os.path.join(current_path +'/Example_ML_Regression_Model.pmml')
example_ML_model = Model.fromFile(example_ML_model_file_path)