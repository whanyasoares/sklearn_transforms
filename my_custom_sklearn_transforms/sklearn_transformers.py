from sklearn.base import BaseEstimator, TransformerMixin


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
