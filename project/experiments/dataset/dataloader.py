import pandas as pd


class DataLoader():
    """
    
    """
    
    def __init__(self, dataset_manager):
        # The dataset manager has access to dataset details from the configuration file
        self.dataset_manager = dataset_manager
        self.target_column = self.dataset_manager.target_column
        self.target_categories = self.dataset_manager.target_categories 

        # For common OHE when applied in different experiments
        from dask_ml.preprocessing import OneHotEncoder
        self.encoder =  OneHotEncoder(sparse=False)
        
    """
    Regular Dask loader
    """
    def get_train(self, dask=True):
        """
        # TODO consider randomized distributed sampling options 
        (might be able to just do this in Dask though)
        https://github.com/tcvrick/COMP551-Final-Project/blob/master/project_src/experiments/datasets/exp0_mnist.py
        https://docs.databricks.com/applications/deep-learning/data-prep/petastorm.html
        """
        train_ddf = self.dataset_manager.get_training_set()

        return train_ddf


    def get_test(self, dask=True):
        """
        Dask test loader with preprocessing on X_test
        """
        test_ddf = self.dataset_manager.get_test_set()

        return test_ddf

    
    """
    Dask loader with OHE
    # TODO reset ddf handle after transformation to pandas
    """
    def get_train_ohe(self, categorical_columns_to_ohe, datetime_columns_to_ohe, dask=True):
        """
        Dask test loader with preprocessing on X_train
        """
        train_ddf = self.dataset_manager.get_training_set()
        return self.data_prep_ohe(train_ddf, categorical_columns_to_ohe, datetime_columns_to_ohe, dask)
    
    
    def get_test_ohe(self, categorical_columns_to_ohe, datetime_columns_to_ohe, dask=True):
        """
        Dask test loader with preprocessing on X_train
        """
        test_ddf = self.dataset_manager.get_test_set()
        return self.data_prep_ohe(test_ddf, categorical_columns_to_ohe, datetime_columns_to_ohe, dask)
        


    def data_prep_ohe(self, ddf, categorical_columns_to_ohe, datetime_columns_to_ohe, dask):
        
        # Make binary targets from config if categories are specified
        if self.target_categories is not None:
            ddf = self.make_binary_target(ddf, self.target_column, self.target_categories)

        # Make OHE representation, tossing other categoricals
        # TODO check if this messes with target category (e.g., removes it)
        ddf = self.process_ohe(ddf, 
                            categorical_columns_to_ohe, 
                            datetime_columns_to_ohe)
        # Get fixed pandas sample from a number of dataframe partitions
        if not dask:
            ddf = self.convert_to_pandas(ddf, 10)
        
        # Convert to X, y
        y = ddf[self.target_column]
        X = ddf.drop(self.target_column, axis=1)
        return X, y


    
#     def get_X_y(self, ddf):
#         """
        
#         """
#         target_column = self.dataset_manager.target_column
#         binary_target_tup = self.dataset_manager.target_categories 
        
#         if binary_target_tup is not None:
#             self.make_binary_target(ddf, target_column, binary_target_tup)
#         y_ddf = ddf[target_column]
#         X_ddf = ddf.drop(target_column, axis=1)
        
#         return X_ddf, y_ddf
        
        

    def process_ohe(self, ddf, categorical_columns_to_ohe, datetime_columns_to_ohe):
        """
        Preprocessing step for OHE of dask ddf
        """
        import project.utils.preprocessing.preprocessing_pipelines as preprocessing

        ddf = preprocessing.ohe_preprocessing_pipeline(self.encoder, 
                                         ddf, 
                                         categorical_columns_to_transform = categorical_columns_to_ohe, 
                                         datetime_columns_to_transform = datetime_columns_to_ohe)

        return ddf


    def make_binary_target(self, ddf, target_column, binary_target_tup):
        """
        Takes a dask dataframe to change categorical targets into a binary target
        """

        print('Targets are: 0-' + binary_target_tup[0] + ' 1-'+ binary_target_tup[1])
        
        # Filters away target categories that are not of interest
        filtered_ddf = ddf[(ddf[target_column] == binary_target_tup[0]) | (ddf[target_column] == binary_target_tup[1])]
        
        # Replaces the string representation of the targets with a numerical representation
        def replace(df: pd.DataFrame) -> pd.DataFrame:
            return df.replace({target_column: [binary_target_tup[0], binary_target_tup[1]]},
                             {target_column: [0, 1]})
        filtered_ddf = filtered_ddf.map_partitions(replace)
        filtered_ddf[target_column] = filtered_ddf[target_column].astype(int)
        
        # Check
#         print(filtered_ddf[target_column].unique().compute())
        
        return filtered_ddf          
        

    def convert_to_pandas(self, ddf, partitions_sample):
        """
        Conversion to pandas/moving dataframe to memory for library support
        """
        
        import project.utils.preprocessing.dask_to_pandas as dtp
        return dtp.dask_ddf_to_df(ddf, partitions_to_concat=partitions_sample)
        
#         X, y = dtp.dask_Xy_to_df(X_ddf, y_ddf, self.dataset_manager.target_column, partitions_to_concat)

#         return X, y
    

    def fill_nulls(df):
        """
        """
        # Fill null values
        #     categorical_cols = X.select_dtypes(include=['category'])
        #     for category in categorical_cols:
        #         X[category].add_categories('Unknown', inplace=True)
        #     X[categorical_cols].fillna('Unknown', inplace=True)

        df.fillna(0, inplace=True)


