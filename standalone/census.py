import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def main():

    #
    # Reads the CSV file containing the data
    ds = pd.read_csv("data/Census01.csv")

    #
    # Defines the explanatory variables per categories (nominal and numerical, the latter needed encoding)
    numerical_variables = ["age","fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    nominal_variables = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    all_variables = get_all_variables(numerical_variables, nominal_variables)
    print(all_variables)

    #
    # Define the target variable
    target_variable = "class"

    #
    # Prepares the dataset
    prepare_data_set(ds,nominal_variables)

    #
    # Splits the data set into an estimation dataset and a validation dataset (actually several splits are possible)
    split = cross_validation.ShuffleSplit(ds.shape[0], n_iter=1, random_state=1, test_size=.33)

    algorithm = RandomForestClassifier(n_estimators=10)
    # algorithm = LinearRegression()

    for estimation_indices, validation_indices in split:

        #
        # Extracts the records based on the split indices
        estimation_records = ds[all_variables].iloc[estimation_indices,:]
        estimation_target = ds[target_variable].iloc[estimation_indices]

        #
        # Computes the model
        algorithm.fit(estimation_records, estimation_target)

        #
        # Estimates the model performance by applying it to the validation dataset
        validation_predictions = algorithm.predict(ds[all_variables].iloc[validation_indices,:])

        validation_predictions[validation_predictions > .5] = 1
        validation_predictions[validation_predictions <= .5] = 0
        acc = accuracy_score(ds[target_variable].iloc[validation_indices], validation_predictions)
        print("Accuracy : {}".format(acc))


def prepare_data_set(ds, nominal_variables):
    for nominal_variable in nominal_variables:
        encode_nominal(ds, nominal_variable)


def get_all_variables(numerical_variables, nominal_variables):
    result = []
    result = result + numerical_variables
    for nominal_variable in nominal_variables:
        result.append(get_encoded_variable_name(nominal_variable))
    return result


def encode_nominal(ds, variable_name):

    #
    # Adds a new variable the name of which is derived from the name of the nominal variable
    encoded_variable_name = get_encoded_variable_name(variable_name)

    #
    # Uses the "ordinal" of the variable values as the encoded form for the nominal variable
    values = ds[variable_name].unique()
    encoded_value = 0
    for value in values:
        ds.loc[ds[variable_name] == value, encoded_variable_name] = encoded_value
        encoded_value += 1


def get_encoded_variable_name(variable_name):
    encoded_variable_name = "___" + "encoded" + "_" + variable_name
    return encoded_variable_name

main()
