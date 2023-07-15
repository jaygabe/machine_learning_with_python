

from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer


###########################################################################
# UPLOAD DATA - UPLOAD DATA - UPLOAD DATA - UPLOAD DATA - UPLOAD DATA - UPL
###########################################################################
# Required modules:
from pathlib import Path
import urllib.request
import tarfile
import pandas as pd
# Create a function called load_housing_data
# Use built-in Path function to import data from the datasets directory
# Store the Path(file) inside a variable called tarball_path
# This enables us to see if this is a file
# If it is not a file
# Use the Path function to create a directory called "datasets"
# Set parents=True to ensure all directories required are created
# Set exist_ok=True to ensure we don't get error if already exists
# Retrieve the data from the URL where the data is stored
# Pass URL and tarball_path
# Open the data file and name it "housing_tarball"
# Extract all data into "datasets" directory we just created
# Handle the error (ENSURE WE ALWAYS RECEIVE DATA)
# Use Pandas to return a Dataframe
###########################################################################
# UPLOAD DATA - UPLOAD DATA - UPLOAD DATA - UPLOAD DATA - UPLOAD DATA - UPL
###########################################################################


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        try:
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                print("tarball:", housing_tarball.getnames())
                housing_tarball.extractall(path="datasets")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None  # Handle error another way to guarantee data
    return pd.read_csv(Path("datasets/housing/housing.csv"))
###########################################################################
###########################################################################
###########################################################################


###########################################################################
# VIEW DATA - VIEW DATA - VIEW DATA - VIEW DATA - VIEW DATA - VIEW DATA - V
###########################################################################
# Execute the load_housing_data function to store data in housing variable
# Display first five rows with the .head() method
# Display summary of Dataframe with the .info() method
# Display descriptive statistics with the .describe() method
# Find details about vectors based on ocean proximity
###########################################################################
# VIEW DATA - VIEW DATA - VIEW DATA - VIEW DATA - VIEW DATA - VIEW DATA - V
###########################################################################
housing = load_housing_data()
print("Head:")
print(housing.head())
print("Info:")
print(housing.info())
# Standard Deviation, Mean, Quartiles, Min, Max, and more
print("Describe:")
print(housing.describe())
print(housing["ocean_proximity"].value_counts())
###########################################################################
###########################################################################
###########################################################################


###########################################################################
# SAVE A FIGURE - SAVE A FIGURE - SAVE A FIGURE - SAVE A FIGURE - SAVE A FI
###########################################################################
# Required modules:
# from pathlib import Path (Already Declared)
import matplotlib.pyplot as plt
# Create a new path object of "/images/end_to_end_project"
# Create the new directories
# Create function to save figure/plot
# Create the path to save the image with the fig_id and extension as file
# if tight_layout is set to true, execute tight_layout on plot
# Save figure to file with plot.savefig() method
# Set default plot font size to 14
# Set default value for labels and titles to 14
# Set the font size for the legend to 14
# Set font size on ytick and xtick to 14
# Create a histogram with the housing Dataframe
# Save housing histogram with our save_fig function
# Display housing histogram with plot.show() method
###########################################################################
# SAVE A FIGURE - SAVE A FIGURE - SAVE A FIGURE - SAVE A FIGURE - SAVE A FI
###########################################################################
IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.fig_extension"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")
plt.show()
###########################################################################
###########################################################################
###########################################################################

###########################################################################
###########################################################################
###########################################################################
# CREATE TEST SET - CREATE TEST SET - CREATE TEST SET - CREATE TEST SET - C
###########################################################################
###########################################################################
###########################################################################

###########################################################################
# RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - R
###########################################################################
# Required modules:
import numpy as np
from zlib import crc32
# Set random seed to 42, so we always get the same training and test set
# Use the number of elements in the Dataframe create permutation of indices
# Find out what our test set size is with the len of data * test_ratio
# Use test_set_size to create list of test_indices with shuffled_indices
# Use test_set_size to create list of train_indices with shuffled_indices
# Return two Dataframes: training set and test set
# Execute function to create train_set and test_sets
# Print the length of each set for confirmation
###########################################################################
# RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - R
###########################################################################

# np.random.seed(42)
#
#
# def shuffle_and_split_data(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
#
# train_set, test_set = shuffle_and_split_data(housing, 0.2)
# print("Length of training set:", len(train_set))
# print("Length of test set:", len(test_set))

###########################################################################
# RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - RANDOM - R
###########################################################################

###########################################################################
# HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - H
###########################################################################
# Create a function that determines whether an instance is in the test set
# If the hash is less than 858993459.2, return false, otherwise true
# Create an object of ids that are present in test set
# Use id series to apply a Lambda function and retrieve a true/false series
# Return training set and test set with in_test_set and negated in_test_set
# Use .reset_index on housing Dataframe to ensure index field exists
# Execute split_data_with_id_hash with housing_with_id, 0.2 ratio and index
# Create new id column by multiplying long value by 1000 and adding lat
# Execute split_data_with_id_hash with housing_with_id, 0.2 ratio and id
###########################################################################
# HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - H
###########################################################################


# def is_id_in_test_set(identifier, test_ratio):
#     return crc32(np.int64(identifier)) < test_ratio * 2**32
#
#
# def split_data_with_id_hash(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]
#
#
# housing_with_id = housing.reset_index()  # adds an `index` column
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
# print("Hashed Test Set (Index):", len(test_set))
# print("Hashed Training Set (Index): ", len(train_set))
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
# print("Hashed Test Set (Id):", len(test_set))
# print("Hashed Training Set (Id):", len(train_set))

###########################################################################
# HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - HASHED - H
###########################################################################

###########################################################################
# SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN
###########################################################################
# Required Modules:
from sklearn.model_selection import train_test_split
# Use train_test_set from scikit learn to create test and training sets
# Find out how many instances have a null value for total_bedrooms
###########################################################################
# SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN
###########################################################################

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("Scikit Learn Test Set:", len(test_set))
print("Scikit Learn Training Set:", len(train_set))

sumOfNullFeatures = test_set["total_bedrooms"].isnull().sum()
print("sumOfNullFeatures: ", sumOfNullFeatures)

###########################################################################
# SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN - SCIKIT-LEARN
###########################################################################

###########################################################################
# Stratified Sampling - # Stratified Sampling - # Stratified Sampling - # S
###########################################################################
# Required Modules:
from scipy.stats import binom
###########################################################################
# Stratified Sampling - # Stratified Sampling - # Stratified Sampling - # S
###########################################################################

sample_size = 1000
ratio_female = 0.511
proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print("Probability of incorrect ratio: ", proba_too_small + proba_too_large)

###########################################################################
# Stratified Sampling - # Stratified Sampling - # Stratified Sampling - # S
###########################################################################

###########################################################################
###########################################################################
###########################################################################
# CREATE TEST SET - CREATE TEST SET - CREATE TEST SET - CREATE TEST SET - C
###########################################################################
###########################################################################
###########################################################################




# housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# # housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# # plt.xlabel("Income category")
# # plt.ylabel("Number of districts")
# # # save_fig("housing_income_cat_bar_plot")  # extra code
# # plt.show()
#
# """ from sklearn.model_selection import StratifiedShuffleSplit """
# splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#     strat_train_set_n = housing.iloc[train_index]
#     strat_test_set_n = housing.iloc[test_index]
#     strat_splits.append([strat_train_set_n, strat_test_set_n])
#
# strat_train_set_n, strat_test_set_n = strat_splits[0]
# strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#
# """
#     EXPLORING THE TRAINING DATA
# """
# print("Training Data: ", strat_train_set)
# print("Test Data: ", strat_test_set)
#
# # Make a copy of the original so we can refer to it later
# housing_training_copy = strat_train_set.copy()
#
# # housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
# # plt.show()
# #
# # housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# # plt.show()
#
# """
#     Plot data by latitude and longitude
#     Use size of circle to determine population
#     Use color of circle to determine house value
# """
# # housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population", c="median_house_value", cmap="jet", colorbar=True, sharex=False, figsize=(10, 7))
# # plt.show()
#
# """
#     Find a correlations between the data
#     Only found one between median_house_value and median_income (linear)
#     Printed on command line
# """
# # numeric_columns = housing.select_dtypes(include={np.number})
# # corr_matrix = numeric_columns.corr()
# # print(corr_matrix["median_house_value"].sort_values(ascending=False))
# # # # Visualized with a graph ^^^
# # attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# # scatter_matrix(housing[attributes], figsize=(12, 8))
# # plt.show()
#
# """
#     Plot median_income against median_house_value to see linear relationship
# """
# # housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
# # plt.show()
#
# """
#     Use the data to:
#     1) Find rooms per house by dividing total_rooms by households
#     2) Find bedroom ration by dividing total_bedrooms by total_rooms
#     3) Find people_per_house by dividing population by households
# """
# housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
# housing["people_per_house"] = housing["population"] / housing["households"]
#
# numeric_columns = housing.select_dtypes(include={np.number})
# corr_matrix = numeric_columns.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
#
# """
#     Preparing the Data for Machine Learning Algorithms
# """
# housing = strat_train_set.drop("median_house_value", axis=1)
# housing_labels = strat_train_set["median_house_value"].copy()
# print(housing)
# print(housing_labels)
#
# """
#     Clean the data
# """
# # housing.dropna(subset=["total_bedrooms"], inplace=True) # Get rid of corresponding districts
# housing.drop("total_bedrooms", axis=1) # Get rid of the whole attribute
#
# # CHOOSE OPTION 3
# median = housing["total_bedrooms"].median() # Set missing value as median
# housing["total_bedrooms"].fillna(median, inplace=True)
#
# # CHOOSE OPTION 3 BUT USE BUILT-IN SCIKIT-LEARN CLASS
# """
#     Using scikit-learn class allows us to impute missing values on the training set,
#     BUT ALSO on the validation and test set, and any new data added to the set
# """
# imputer = SimpleImputer(strategy="median")
# # attain only numerical columns
# housing_num = housing.select_dtypes(include=[np.number])
# # Fit imputer instance to the training data using the fit() method
# imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)
# X = imputer.transform(housing_num)
#
# # Wrap X in a Dataframe and recover the column names and index from housing_num
# housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
#
# """
#     Handling Text and Categorical Attributes
# """
# housing_cat = housing[["ocean_proximity"]]
# print(housing_cat.head(8))
# #
# # # Convert these categories from text to numbers
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# print(ordinal_encoder.categories_)
#
# """
#     Use one-hot encoding to prevent errors with categorical data
# """
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)
# print(housing_cat_1hot.toarray())
# print(cat_encoder.categories_)
# # Print the categories with one-hot representation
# df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
# print(pd.get_dummies(df_test))
#
# print(cat_encoder.transform(df_test))
#
# df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
# pd.get_dummies(df_test_unknown)
#
# print(cat_encoder.feature_names_in_)
# print(cat_encoder.get_feature_names_out())
#
# # df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown), columns=cat_encoder.get_feature_names_out(),
# #                          index=df_test_unknown.index)
#
# """
#     Feature Scaling and Transformation
# """
# # Min-max Scaling
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
# housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# print(housing_num_min_max_scaled)
#
# #Standardization
# std_scaler = StandardScaler()
# housing_num_std_scaled = std_scaler.fit_transform(housing_num)
# print(housing_num_std_scaled)
#
# age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
# print("RBF Kernel: ", age_simil_35)
#
# # Create scatterplot of rbf results
# plt.figure(figsize=(10, 6))
# plt.scatter(housing["housing_median_age"], age_simil_35)
# plt.xlabel('Housing Median Age')
# plt.ylabel('RBF Kernel Similarity with Age 35')
# plt.title('RBF Kernel Similarity vs. Housing Median Age')
# plt.grid(True)
# plt.show()
#
# target_scaler = StandardScaler()
# scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
# model = LinearRegression()
# model.fit(housing[["median_income"]], scaled_labels)
# some_new_data = housing[["median_income"]].iloc[:5] # pretend this is new data
# scaled_predictions = model.predict(some_new_data)
# predictions = target_scaler.inverse_transform(scaled_predictions)
# print("Predictions 1:", predictions)
#
#
# model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
# model.fit(housing[["median_income"]], housing_labels)
# predictions = model.predict(some_new_data)
# print("Predictions 2:", predictions)
#
# log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
# log_pop = log_transformer.transform(housing[["population"]])
# print(log_pop)