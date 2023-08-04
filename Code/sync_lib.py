import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from nltk.util import ngrams
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.distances import dtw_distance
import matplotlib.pyplot as plt


def substitutes_empty_values_by_ones_behind(df):
    for col in df.columns:
        for row in df.index[1:]: #what if the first row is empty? maybe we should remove it
            if pd.isna(df[col][row]):
                value_to_update = df.at[row - 1,col]
                df.at[row,col] = value_to_update
    return df

#changes the frequency of a dataframe by turning every row that is in the desired freq interval to one single row corresponding to the moving averadge
def ajust_df_frequency_by_mv_avg(df_wrong_freq, desired_freq):
    
    initial_time_value = pd.to_datetime(df_wrong_freq["Time"].iloc[0])
    initial_time_value = initial_time_value.replace(second = (initial_time_value.second // desired_freq) * desired_freq)
    
    time_interval_col = pd.date_range(start=initial_time_value, periods= df_wrong_freq.shape[0] , freq= f"{desired_freq}S")
    
    list_columns = list(df_wrong_freq.columns)
    list_columns.remove("Time")

    values_mv_avg = np.zeros((df_wrong_freq.shape[0],df_wrong_freq.shape[1] - 1))
    df_mv_avg = pd.DataFrame(data= values_mv_avg,columns=list_columns)
    df_mv_avg["Time_Interval"] = time_interval_col
    columns_without_time = list_columns
    iterator_mv_avg = 0
    num_cols_interval = 0
    timestamp_mv_avg = initial_time_value

    #fix date format, pandas is changing the order of month and day for some reason
    num_col_output = 0
    for row in tqdm(df_wrong_freq.index):
        #print(df_wrong_freq.at[row,"Time"] , timestamp_mv_avg + pd.Timedelta(seconds=desired_freq))
        if df_wrong_freq.at[row,"Time"] > timestamp_mv_avg + pd.Timedelta(seconds=desired_freq): #df_mv_avg.at[iterator_mv_avg + 1,"Time_Interval"]
            for col in columns_without_time:
                df_mv_avg.at[iterator_mv_avg, col] /= num_cols_interval  
            df_mv_avg.at[iterator_mv_avg, "Time_Interval"] = timestamp_mv_avg
            
            num_cols_interval = 0
            iterator_mv_avg += 1
            measure_time = df_wrong_freq.at[row,"Time"]
            timestamp_mv_avg = measure_time.replace(second = (measure_time.second // desired_freq) * desired_freq)
            num_col_output += 1
            

        num_cols_interval += 1 

        for col in columns_without_time:
            df_mv_avg.at[iterator_mv_avg, col] += df_wrong_freq.at[row,col]
        

    for col in columns_without_time:
        df_mv_avg.at[iterator_mv_avg, col] /= num_cols_interval  
    df_mv_avg.at[iterator_mv_avg, "Time_Interval"] = timestamp_mv_avg
    df_mv_avg = df_mv_avg.iloc[:iterator_mv_avg+1,:]

    

    return df_mv_avg
    #'''
        
def add_prefixes_in_columns_names(df,df_prefix,col_in_common = "Time_Interval"):
    columns_df_renamed = []
    list_columns_df = list(df.columns)
    list_columns_df.remove(col_in_common)
    for col in list_columns_df:
        columns_df_renamed.append(df_prefix + col)
    columns_df_renamed.append(col_in_common)  
    df.columns = columns_df_renamed
    return df
    
#Renames the two input dfs according to the above function and merges by concatenating all rows in wich the value of the column col_in_common belongs to both dfs 
def merge_on_comon_col_adding_prefixes_to_col_names(df1,df2,df1_prefix="",df2_prefix="",col_in_common = "Time_Interval"):
    df1_renamed = add_prefixes_in_columns_names(df1,df1_prefix,col_in_common)
    df2_renamed = add_prefixes_in_columns_names(df2,df2_prefix,col_in_common)
    return df1_renamed.merge(df2_renamed,how="inner",on=col_in_common)


#TODO: make this function cleaner
#removes nan values or puts the previous column value according to the rule you sent me on whtasapp 
def remove_nan_dfs(df1_intersect,df2_intersect,window_size= 5,max_phase_dif = 8,time_col = "Time"):
    # Loop through the DataFrame with a sliding window
    indexes_to_drop_df1 = []
    indexes_to_drop_df2 = []
    i1 = 0
    i2 = 0
    while (i1 < ( len(df1_intersect) - window_size + 1)) and i2 < (( len(df2_intersect) - window_size + 1)):
        df1_time = df1_intersect[time_col][i1]
        df2_time = df2_intersect[time_col][i2] 

        window_1 = df1_intersect[i1:i1 + window_size]
        window_2 = df2_intersect[i2:i2 + window_size]

        indexes_to_drop_df1 = []
        indexes_to_drop_df2 = []

        if np.abs((df1_time - df2_time).total_seconds()) > max_phase_dif:
            print(df2_intersect.loc[i2] ,df2_intersect.loc[i2].isnull().any())

            if df1_time > df2_time:
                if df2_intersect.loc[i2].isnull().any():
                    indexes_to_drop_df2.append(i2)
                i2 += 1
            else:
                if df1_intersect.loc[i1].isnull().any():
                    indexes_to_drop_df1.append(i1)
                i1 += 1
            continue
        window_1 = df1_intersect[i1:i1 + window_size]
        window_2 = df2_intersect[i2:i2 + window_size]
        # Check if all rows in the window have at least one NaN value in both dfs
        if window_1.isnull().any(axis=1).all() and window_2.isnull().any(axis=1).all():
            print("dropping")
            indexes_to_drop_df1 += list(window_1.index)
            indexes_to_drop_df2 += list(window_2.index)
            
        i1 += 1
        i2 += 1
    
    print(indexes_to_drop_df1, indexes_to_drop_df2)
    df1_intersect.drop(axis=0, index= indexes_to_drop_df1 ,inplace = True)
    df2_intersect.drop(axis=0, index= indexes_to_drop_df2 ,inplace = True)
        
    df1_without_nan = df1_intersect.reset_index(drop=True)
    df2_without_nan = df2_intersect.reset_index(drop=True)
    return df1_without_nan,df2_without_nan

# test if values make sense after normalyzing
def separates_non_numerical_columns(df: pd.DataFrame):
    non_numerical_columns = dict()

    for col_name, col_data in df.items():
        try: 
            df[col_name] = df[col_name].astype(float)
        except:
            non_numerical_columns[col_name] = col_data
            df.drop([col_name], axis=1, inplace=True)
    return df, non_numerical_columns

def normalyzes_numerical_values_df(df: pd.DataFrame):
    numerical_df, non_numerical_columns = separates_non_numerical_columns(df)
    values_df = numerical_df.values
    columns_df = numerical_df.columns

    normalized_values = StandardScaler().fit_transform(values_df)
    normalized_df = pd.DataFrame(data = normalized_values, columns=columns_df)
    
    for key, value in non_numerical_columns.items():
        normalized_df[key] = value
    return normalized_df
#test it also

def reduces_dimensionality_by_PCA(normalized_df: pd.DataFrame, num_dim_reductiopn: int):
    normalized_num_df, non_numerical_columns = separates_non_numerical_columns(normalized_df)
    values_df = normalized_num_df.values
    pca_instance = PCA(n_components=num_dim_reductiopn)
    dimension_reducted_values = pca_instance.fit_transform(values_df)
    columns_name_after_reduction = generate_eigenvalue_named_columns(num_dim_reductiopn)
    dimension_reducted_df = pd.DataFrame(data = dimension_reducted_values, columns=columns_name_after_reduction)
    print('Explained variation per principal component: {}'.format(pca_instance.explained_variance_ratio_))


    for key, value in non_numerical_columns.items():
        dimension_reducted_df[key] = value
    
    return dimension_reducted_df

def generate_eigenvalue_named_columns(number_columns):
    return [f"eigenvalue_{i+1}"  for i in range(number_columns)] 

def compensates_drift_with_PCA(normalized_df: pd.DataFrame):
    print(normalized_df.columns)
    time_column = normalized_df['Time']
    dimension_reducted_df = reduces_dimensionality_by_PCA(normalized_df, 2)
    most_significant_eigenvector = dimension_reducted_df[dimension_reducted_df.columns[0]]
    
    # check if this transpose is correct, maybe the shapes are should be different
    tst = np.matmul(normalized_df.T, most_significant_eigenvector)
    print("tst.shape, most_significant_eigenvector.shape, normalized_df.shape", tst.shape, most_significant_eigenvector.shape, normalized_df.shape)
    # tst_matrix = np.reshape(tst, (-1, 1))
    # most_significant_eigenvector_matrix = np.reshape(most_significant_eigenvector, (1, -1))
    
   
    corrected = normalized_df - np.matmul(most_significant_eigenvector[:, np.newaxis], tst.T[np.newaxis,:])
    drift_conpensated_df = np.abs(corrected)
    drift_conpensated_df["Time"] = time_column

    return drift_conpensated_df



# TODO:
# 1. wrap around the pd.DataFrame with a class that represents the kind of table we are dealing with
# 2. rewrite this library  with classes maybe two classes one for syncronization and one for other things such drift compensation
# 3. 