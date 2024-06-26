# TCOCNN
Extensive research has been conducted on the TCOCNN, a neural network specializing in analyzing MOS gas sensor data for air quality assessments. This cutting-edge technology has the potential to revolutionize the air quality measurement process and lead to significant advancements in public health.

# Example (data used in https://www.mdpi.com/2073-4433/13/10/1614)
Download the dataset from https://zenodo.org/record/6821340.
This dataset contains a small dataset to test transfer learning.

Create a new folder within the toolbox named Data and store the downloaded fullData there.

Execute the PrepareDataset.ipynb File. 

Execute the testBuiltModel.

From here on, every method can be tested.

# Example 2 (data used in https://www.mdpi.com/2073-4433/12/11/1487)
Another dataset is available at https://zenodo.org/records/4593853.
This dataset contains a small dataset to test for drift and field tests.

Create a new folder within the toolbox named Field and store the downloaded data there.

Execute the PrepareDatasetField.ipynb File. 

Execute the testBuiltModelField.

From here on, every method can be tested.

# Example 3 (data used in https://www.mdpi.com/2073-4433/12/11/1487)

The scripts for the evaluation for paper 3 (https://www.mdpi.com/2073-4433/14/7/1123) are available, and the data will be available soon.

# Requierments
The TCOCNN was built with tensorflow 2.6, keras 2.6, scikit-learn, matplotlib, numpy, skopt, sklearn.