"""
A script to clean data.
"""

# imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import logging


class dataCleaner():
    """
    A data cleaner class.
    """
    def __init__(self, fromThe: str) -> None:
        """
        The data cleaner initializer

        Parameters
        =--------=
        fromThe: string
            The file importing the data cleaner

        Returns
        =-----=
        None: nothing
            This will return nothing, it just sets up the data cleaner
            script.
        """
        try:
            # setting up logger
            self.logger = self.setup_logger('../logs/cleaner_root.log')
            self.logger.info('\n    #####-->    Data cleaner logger for ' +
                             f'{fromThe}    <--#####\n')
            print('Data cleaner in action')
        except Exception as e:
            print(e)

    def setup_logger(self, log_path: str) -> logging.Logger:
        """
        A function to set up logging

        Parameters
        =--------=
        log_path: string
            The path of the file handler for the logger

        Returns
        =-----=
        logger: logger
            The final logger that has been setup up
        """
        try:
            # getting the log path
            log_path = log_path

            # adding logger to the script
            logger = logging.getLogger(__name__)
            print(f'--> {logger}')
            # setting the log level to info
            logger.setLevel(logging.INFO)
            # setting up file handler
            file_handler = logging.FileHandler(log_path)

            # setting up formatter
            formatter = logging.Formatter(
                "%(levelname)s : %(asctime)s : %(name)s : %(funcName)s " +
                "--> %(message)s")

            # setting up file handler and formatter
            file_handler.setFormatter(formatter)
            # adding file handler
            logger.addHandler(file_handler)

            print(f'logger {logger} created at path: {log_path}')
            # return the logger object
        except Exception as e:
            logger.error(e)
            print(e)
        finally:
            return logger

    def remove_unwanted_cols(self, df: pd.DataFrame,
                             cols: list) -> pd.DataFrame:
        """
        A function to remove unwanted columns from a DataFrame

        Parameters
        =--------=
        df: pandas dataframe
            The data frame containing all the data
        cols: list
            The unwanted columns lists

        Returns
        =-----=
        df
            The dataframe rid of the unwanted cols
        """
        try:
            for col in cols:
                df = df[df.columns.drop(list(df.filter(regex = col)))]
                self.logger.info(f'column: {col} removed successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return df        

    def percent_missing(self, df: pd.DataFrame) -> None:
        """
        A function telling how many missing values exist or better still
        what is the % of missing values in the dataset?

        Parameters
        =--------=
        df: pandas dataframe
            The data frame to calculate the missing values from

        Returns
        =-----=
        None: nothing
            Just prints the missing value percentage
        """
        try:
            # Calculate total number of cells in dataframe
            totalCells = np.product(df.shape)

            # Count number of missing values per column
            missingCount = df.isnull().sum()

            # Calculate total number of missing values
            totalMissing = missingCount.sum()

            # Calculate percentage of missing values
            print(f"The dataset contains {round(((totalMissing/totalCells)*100), 10)} % missing values")
            self.logger.info(f"The dataset contains {round(((totalMissing/totalCells)*100), 10)} % missing values")
        except Exception as e:
            self.logger.error(e)
            print(e)

    def fillWithMedian(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        A function that fills null values with their corresponding median
        values

        Parameters
        =--------=
        df: pandas data frame
            The data frame with the null values
        cols: list
            The list of columns to be filled with median values

        Returns
        =-----=
        df: pandas data frame
            The data frame with the null values replace with their
            corresponding median values
        """
        try:
            print(f'columns to be filled with median values: {cols}')
            df[cols] = df[cols].fillna(df[cols].median())
            self.logger.info(f'cols: {cols} filled with median successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return df

    def fillWithMean(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        A function that fills null values with their corresponding mean
        values

        Parameters
        =--------=
        df: pandas data frame
            The data frame with the null values
        cols: list
            The list of columns to be filled with mean values

        Returns
        =-----=
        df: pandas data frame
            The data frame with the null values replace with their
            corresponding mean values
        """
        try:
            print(f'columns to be filled with mean values: {cols}')
            df[cols] = df[cols].fillna(df[cols].mean())
            self.logger.info(f'cols: {cols} filled with mean successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return df

    def fix_outlier(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        A function to fix outliers with median

        Parameters
        =--------=
        df: pandas data frame
            The data frame containing the outlier columns
        column: str
            The string name of the column with the outlier problem

        Returns
        =-----=
        df: pandas data frame
            The fixed data frame
        """
        try:
            print(f'column to be filled with median values: {column}')
            df[column] = np.where(df[column] > df[column].quantile(0.95),
                                  df[column].median(), df[column])
            self.logger.info(f'column: {column} outlier fixed successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return df[column]

    # TODO  : determine which one is better
    def fix_outlier_(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A function to fix outliers

        Parameters
        =--------=
        df:  pandas data frame
            The data frame containing the outlier columns

        Returns
        =-----=
        df:  pandas data frame
            The data frame with the outlier columns fixed
        """
        try:
            self.logger.info('setting up columns to be fixed for outlier')
            # TODO : either pass the outlier columns or checkout the columns
            # list
            column_name = list(df.columns[2:])
            for i in column_name:
                upper_quartile = df[i].quantile(0.75)
                lower_quartile = df[i].quantile(0.25)
                df[i] = np.where(df[i] > upper_quartile, df[i].median(),
                                 np.where(df[i] < lower_quartile,
                                 df[i].median(), df[i]))
            self.logger.info('outliers fixed successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return df

    def choose_k_means(self, df: pd.DataFrame, num: int):
        """
        A function to choose the optimal k means cluster

        Parameters
        =--------=
        df: pandas data frame
            The data frame that holds all the values
        num: integer
            The x scale

        Returns
        =-----=
        distortions and inertias
        """
        try:
            distortions = []
            inertias = []
            K = range(1, num)
            for k in K:
                k_means = KMeans(n_clusters=k, random_state=777).fit(df)
                distortions.append(sum(
                    np.min(cdist(df, k_means.cluster_centers_, 'euclidean'),
                           axis=1)) / df.shape[0])
                inertias.append(k_means.inertia_)
            self.logger.info(f'distortion: {distortions} and inertia:' +
                             f'{inertias} calculated for {num} number of'
                             'clusters successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return (distortions, inertias)

    def computeBasicAnalysisOnClusters(self, df: pd.DataFrame,
                                       cluster_col: str, cluster_size: int,
                                       cols: list) -> None:
        """
        A function that gives some basic description of the 3 clusters

        Parameters
        =--------=
        df: pandas data frame
            The main data frame containing all the data
        cluster_col: str
            The column name holding the cluster values
        cluster_size: integer
            The number of total cluster groups
        cols: list
            The column list on which to provide description

        Returns
        =-----=
        None: nothing
            This function only prints out information
        """
        try:
            i = 0
            for i in range(cluster_size):
                cluster = df[df[cluster_col] == i]
                print("Cluster " + (i+1) * "I")
                print(cluster[cols].describe())
                print("\n")
            self.logger.info(f'basic analysis on {cluster_size} clusters' +
                             'computed successfully')
        except Exception as e:
            self.logger.error(e)
            print(e)



    # new additions
    # TODO: add try, except finally
    # TODO: add comment
    # TODO: add logger
    def fix_missing_ffill(self, df, cols):
        for col in cols:
            old = df[col].isna().sum()
            df[col] = df[col].fillna(method='ffill')
            new = df[col].isna().sum()
            if new == 0:
                print(f"{old} missing values in the column {col} have been replaced \
                    using the forward fill method.")
            else:
                count = old - new
                print(f"{count} missing values in the column {col} have been replaced \
                    using the forward fill method. {new} missing values that couldn't be \
                    imputed still remain in the column {col}.")

    def fix_missing_bfill(self, df, cols):
        for col in cols:
            old = df[col].isna().sum()
            df[col] = df[col].fillna(method='bfill')
            new = df[col].isna().sum()
            if new == 0:
                print(f"{old} missing values in the column {col} have been replaced \
                    using the backward fill method.")
            else:
                count = old - new
                print(f"{count} missing values in the column {col} have been replaced \
                    using the backward fill method. {new} missing values that couldn't be \
                    imputed still remain in the column {col}.")

    def missing_values_table(self, df:pd.DataFrame):
        """
        A function to calculate missing values by column
        """
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * mis_val / len(df)

        # dtype of missing values
        mis_val_dtype = df.dtypes

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype],
                                  axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

        # Sort the table by percentage of missing descending and remove columns with no missing values
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,0] != 0].sort_values(
        '% of Total Values', ascending=False).round(2)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")

        if mis_val_table_ren_columns.shape[0] == 0:
            return

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    def fix_missing_value(self, df, cols, value):
        for col in cols:
            count = df[col].isna().sum()
            df[col] = df[col].fillna(value)
            if type(value) == 'str':
                print(f"{count} missing values in the column {col} have been replaced by \'{value}\'.")
            else:
                print(f"{count} missing values in the column {col} have been replaced by {value}.")
 
    def convert_to_string(self, df, columns) -> pd.DataFrame : 
        for col in columns:
            df[col] = df[col].astype("string")
        return df

    def convert_to_numeric(self, df, columns) -> pd.DataFrame:
        for col in columns:
            df[col] = pd.to_numeric(df[col])
        return df

    def convert_to_int(self, df, columns) -> pd.DataFrame:
        for col in columns:
            df[col] = df[col].astype("int64")
        return df

    def convert_to_datetime(self, df, columns) -> pd.DataFrame:
        try:
            for col in columns:
                df[col] = pd.to_datetime(df[col], errors='raise')
                self.logger.info(f'column: {col} successfully changed to datetime')
        except Exception as e:
            self.logger.error(e)
            print(e)
        finally:
            return df

    def multiply_by_factor(df, columns, factor) -> pd.DataFrame:
        for col in columns:
            df[col] = df[col] * factor
        return df

    def show_cols_mixed_dtypes(df):
        mixed_dtypes = {'Column': [], 'Data type': []}
        for col in df.columns:
            dtype = pd.api.types.infer_dtype(df[col])
            if dtype.startswith("mixed"):
                mixed_dtypes['Column'].append(col)
                mixed_dtypes['Data type'].append(dtype)
        if len(mixed_dtypes['Column']) == 0:
            print('None of the columns contain mixed types.')
        else:
            print(pd.DataFrame(mixed_dtypes))