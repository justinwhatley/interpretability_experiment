

class DataLoader():
    """
    
    """
    
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
        
        # For OHE
        from dask_ml.preprocessing import OneHotEncoder
        self.encoder =  OneHotEncoder(sparse=False)

    """
    Regular Dask loader
    """
    def get_train(self):
        """
        # TODO consider randomized distributed sampling options 
        (might be able to just do this in Dask though)
        https://github.com/tcvrick/COMP551-Final-Project/blob/master/project_src/experiments/datasets/exp0_mnist.py
        https://docs.databricks.com/applications/deep-learning/data-prep/petastorm.html
        """
        X_train, y_train = self.dataset_manager.get_training_set()

        return X_train, y_train


    def get_test(self):
        """
        Dask test loader with preprocessing on X_test
        """
        X_test, y_test = self.dataset_manager.get_test_set()

        return X_test, y_test
    

    """
    Dask loader with OHE
    """
    def get_train_ohe(self, categorical_columns_to_ohe, datetime_columns_to_ohe):
        """
        Dask test loader with preprocessing on X_train
        """
        X_train, y_train = self.dataset_manager.get_training_set()

        X_train = self.preprocess(X_train, 
                            categorical_columns_to_ohe, 
                            datetime_columns_to_ohe)

        return X_train, y_train


    def get_test_ohe(self, categorical_columns_to_ohe, datetime_columns_to_ohe):
        """
        Dask test loader with OHE preprocessing on X_ddf
        """
        X_test, y_test = self.dataset_manager.get_test_set()

        X_test = self.preprocess(X_test, 
                        categorical_columns_to_ohe, 
                        datetime_columns_to_ohe)

        return X_test, y_test


    def preprocess(self, ddf, categorical_columns_to_ohe, datetime_columns_to_ohe):
        """
        Preprocessing step for OHE of dask ddf
        """
        import project.src.preprocessing.preprocessing_pipelines as preprocessing


        ddf = preprocessing.ohe_preprocessing_pipeline(self.encoder, 
                                         ddf, 
                                         categorical_columns_to_transform = categorical_columns_to_ohe, 
                                         datetime_columns_to_transform = datetime_columns_to_ohe)


        return ddf





def convert_to_pandas(X_ddf, y_ddf, target, partitions_to_concat):
    """
    Conversion to pandas/moving dataframe to memory for library support
    """
    import project.src.preprocessing.dask_to_pandas as dtp
    X, y = dtp.dask_Xy_to_df(X_ddf, y_ddf, target, partitions_to_concat)

    return X, y

def fill_nulls(df):
    """
    """
    # Fill null values
    #     categorical_cols = X.select_dtypes(include=['category'])
    #     for category in categorical_cols:
    #         X[category].add_categories('Unknown', inplace=True)
    #     X[categorical_cols].fillna('Unknown', inplace=True)

    df.fillna(0, inplace=True)


