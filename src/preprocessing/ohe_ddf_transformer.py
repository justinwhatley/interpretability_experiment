


class TransformerOHE():
    """
    One-hot-encoder transformer to Dask dataframes
    """
    
    def __init__(self, ddf, encoder, categorical_columns_lst = []):
        self.ddf = ddf
        self.encoder = encoder
        self.categorical_columns_lst = categorical_columns_lst
        
    def add_datetime_categories(self, dt_column_lst, hourslot=True, day_of_week=True):
        """
        Extracts an integer representation for hourslot and/of day of week which is
        turned into a categorical column
        """
        for column in dt_column_lst:
            if hourslot:
                self._add_hourslot_categorical(column)
            if day_of_week:
                self._add_day_of_week_categorical(column)
               
            
    def _add_hourslot_categorical(self, dt_column):
        ddf = self.ddf
        new_column_name = dt_column + '_hourslot'
        ddf[new_column_name] = ddf[dt_column].dt.hour
        ddf[new_column_name] = ddf[new_column_name].astype('category')
        self.categorical_columns_lst.append(new_column_name)


    def _add_day_of_week_categorical(self, dt_column):
        ddf = self.ddf
        new_column_name = dt_column + '_day_of_week'
        ddf[new_column_name] = ddf[dt_column].dt.day
        ddf[new_column_name] = ddf[new_column_name].astype('category')
        self.categorical_columns_lst.append(new_column_name)

    
    def fit_transform(self):
        """
        Fits the OHE to the categorical data in the preassigned ddf, saving to the encoder 
        object
        """
        ddf = self.ddf
        categorical_columns_lst = self.categorical_columns_lst

        # Fit
        ddf[categorical_columns_lst] = ddf[categorical_columns_lst].categorize()
        self.encoder.fit(ddf[categorical_columns_lst])

        # Dask ddf transformer
        self.ohe_ddf = self.encoder.transform(ddf[categorical_columns_lst])
        return self.ohe_ddf
