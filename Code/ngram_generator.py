import pandas as pd
import numpy as np

class NGram_Generator:
    def __init__(self, numerical_type: type) -> None:
        self.numerical_type = numerical_type

    def create_2D_ngrams_series_from_1D_series(self,col:pd.Series, ngram_size:int = 30) -> pd.Series:
        result = []
        #print("col",col.size)
        for i in range(col.size - ngram_size + 1):#(col.size // ngram_size ):#
            ngram = pd.Series(col[i:i+ngram_size],dtype= self.numerical_type)#(col[i*ngram_size:(i+1)*ngram_size],dtype= self.numerical_type)#
            ngram_np_array = np.array(ngram,dtype= self.numerical_type)
            result.append(ngram_np_array)

        result = pd.Series(result) #fix the type of the series it is causing a warning
        return result
    

    def create_3D_ngrams_df_from_2D_df(self, original_df: pd.DataFrame , ngram_size: int) -> pd.DataFrame:
       
        # define a function to generate n-grams for each column
        ngram_df_3d = original_df.apply(self.create_2D_ngrams_series_from_1D_series, axis=0, args=(ngram_size,))
        #percentil_95 = int((ngram_df_3d.shape[0] * 95) / 100)
        #DEBUGGING
        if ngram_df_3d.shape[0] == 0:
            print(original_df.shape, ngram_df_3d.shape, ngram_size)
            print("ngram_df_3d.iloc[0]")
        
        # print(" underestanding ngram  " ,ngram_df_3d.shape, type(ngram_df_3d.iloc[0,:]), ngram_df_3d.iloc[0,:].shape, ngram_df_3d.iloc[0,:])
        # this is a hack to return a dataframe with only one row and should be changed by being ajusted by a parameter
        return ngram_df_3d#.iloc[percentil_95:percentil_95+1][:]#.iloc[percentil_95:percentil_95+1][:]