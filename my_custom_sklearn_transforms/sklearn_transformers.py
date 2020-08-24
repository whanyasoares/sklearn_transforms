from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class DuplicateColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column, new_column):
        self.column = column
        self.new_column = new_column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data[self.new_column] = data[self.column]
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data
    
class ReplaceImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column_replace, column_compare, value_compare, replacer):
        self.column_replace = column_replace
        self.column_compare = column_compare
        self.value_compare = value_compare
        self.replacer = replacer

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()

        data.loc[data[self.column_compare] > self.value_compare, self.column_replace] = self.replacer
        data[self.column_replace].fillna(data[self.column_replace].median(), inplace=True)

        return data
    
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)    
