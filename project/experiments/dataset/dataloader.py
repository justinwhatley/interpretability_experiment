import pandas as pd



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
        train_ddf = self.dataset_manager.get_training_set()

        return train_ddf


    def get_test(self):
        """
        Dask test loader with preprocessing on X_test
        """
        test_ddf = self.dataset_manager.get_test_set()

        return test_ddf
    

    """
    Dask loader with OHE
    """
    def get_train_ohe(self, categorical_columns_to_ohe, datetime_columns_to_ohe):
        """
        Dask test loader with preprocessing on X_train
        binary_target_tup, e.g., ('Normal', 'Blocked')
        """
        X_train, y_train = self.get_train_X_y()
            
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

    
    def get_train_X_y(self):
        """
        
        """
        train_ddf = self.dataset_manager.get_training_set()
        target_column = self.dataset_manager.target_column
        binary_target_tup = self.dataset_manager.target_categories 
        
        if binary_target_tup is not None:
            self.make_binary_target(train_ddf, target_column, binary_target_tup)
        y_ddf = train_ddf[target]
        X_ddf = train_ddf.drop(target, axis=1)
        
        return X_ddf, y_ddf
        
        

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


    def make_binary_target(self, ddf, target_column, binary_target_tup):
        """
        Takes an X dataframe and y series to produce a binary target
        """

        print('Targets are: 0-' + binary_target_tup[0] + ' 1-'+ binary_target_tup[1])

#         filtered_ddf = ddf[(ddf[target_column] == ctrl_targ_tup[0]) | (ddf[target_column] == ctrl_targ_tup[1])]
        
        filtered_ddf = ddf[(ddf[target_column] == binary_target_tup[0]) | (ddf[target_column] == binary_target_tup[1])]


#         filtered_ddf = filter_categories(ddf, binary_target_tup)

    #     y = filtered_ddf[target_column].map({"Normal":0, "Blocked":1, "Dropped":2, "Non-progressed":3}).astype(int)

        # Only works for Pandas
        # y = filtered_ddf[target_column].map({ctrl_targ_tup[0]:0, ctrl_targ_tup[1]:1}).astype(int)
        
#         y = filtered_ddf[target_column].map({ctrl_targ_tup[0]:0, ctrl_targ_tup[1]:1}).astype(int)
#         y = replace(filtered_ddf[target_column], ctrl_targ_tup).astype(int)
        ddf = ddf.map_partitions(replace(target_column, ctrl_targ_tup))

        ddf = ddf.replace({target_column: [ctrl_targ_tup[0], ctrl_targ_tup[1]]},
                         {target_column: ['0', '1']}).astype(int)
    #     print(y.unique())

        return X, y
    
    
    def filter_categories(ddf, ctrl_targ_tup):
        return ddf[(ddf[target_column] == ctrl_targ_tup[0]) | (ddf[target_column] == ctrl_targ_tup[1])]               
        
    
#     def replace(ddf: pd.DataFrame, ctrl_targ_tup) -> pd.DataFrame:
#         """
#         """
#         return ddf.replace({target_column: [ctrl_targ_tup[0], ctrl_targ_tup[1]]},
#                          {target_column: ['0', '1']}).astype(int)



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


