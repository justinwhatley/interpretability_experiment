def add_datetime_cat(ddf, datetime_columns_to_transform, hourslot=True, day_of_week=True):
    """
    For direct transformation without saving the object
    """
    datetime_converter = DatetimeConverter()
    ddf, new_categorical_columns = datetime_converter.add_datetime_categories(ddf, 
                                                                   datetime_columns_to_transform, 
                                                                   hourslot=hourslot, 
                                                                   day_of_week=day_of_week)
    return ddf, new_categorical_columns


class DatetimeConverter():

    def __init__(self):
        self.categorical_columns_lst = []
    
    def add_datetime_categories(self, ddf, dt_column_lst, hourslot=True, day_of_week=True):
        """
        Extracts an integer representation for hourslot and/of day of week which is
        turned into a categorical column
        """
        for column in dt_column_lst:
            if hourslot:
                ddf = self._add_hourslot_categorical(ddf, column)
            if day_of_week:
                ddf = self._add_day_of_week_categorical(ddf, column)
                
        return ddf, self.categorical_columns_lst


    def _add_hourslot_categorical(self, ddf, dt_column):
        new_column_name = dt_column + '_hourslot'
        ddf[new_column_name] = ddf[dt_column].dt.hour
        ddf[new_column_name] = ddf[new_column_name].astype('category')
        self.categorical_columns_lst.append(new_column_name)
        return ddf


    def _add_day_of_week_categorical(self, ddf, dt_column):
        new_column_name = dt_column + '_day_of_week'
        ddf[new_column_name] = ddf[dt_column].dt.dayofweek
        ddf[new_column_name] = ddf[new_column_name].astype('category')
        self.categorical_columns_lst.append(new_column_name)
        return ddf