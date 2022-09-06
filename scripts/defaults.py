"""
A default script for all default variables
"""

# setting up dvc data and folder structure paths
# main data file name
store_data_file = 'store.csv'
train_data_file = 'train.csv'
test_data_file = 'test.csv'

# base repository
repo = '../'

# the path to the data folder
data_path = 'rossmann-store-sales/'

# the local data path
store_local_path = repo + data_path + store_data_file
train_local_path = repo + data_path + train_data_file
test_local_path = repo + data_path + test_data_file

# the path to the data set
store_path = data_path + store_data_file
train_path = data_path + train_data_file
test_path = data_path + test_data_file

# the path to the plots folder
plot_path = 'plots/'
