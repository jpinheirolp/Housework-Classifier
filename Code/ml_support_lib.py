import pandas as pd
import numpy as np
from tqdm import tqdm
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.distances import dtw_distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from Code.sync_lib import generate_eigenvalue_named_columns

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

def compensates_drift_with_PCA(normalized_df: pd.DataFrame):
    print(normalized_df.columns)
    time_column = normalized_df['Time']
    dimension_reducted_df = reduces_dimensionality_by_PCA(normalized_df, 2)
    most_significant_eigenvector = dimension_reducted_df[dimension_reducted_df.columns[0]]
    
    tst = np.matmul(normalized_df.T, most_significant_eigenvector)
    print("tst.shape, most_significant_eigenvector.shape, normalized_df.shape", tst.shape, most_significant_eigenvector.shape, normalized_df.shape)
    # tst_matrix = np.reshape(tst, (-1, 1))
    # most_significant_eigenvector_matrix = np.reshape(most_significant_eigenvector, (1, -1))
    
   
    corrected = normalized_df - np.matmul(most_significant_eigenvector[:, np.newaxis], tst.T[np.newaxis,:])
    drift_conpensated_df = np.abs(corrected)
    drift_conpensated_df["Time"] = time_column

    return drift_conpensated_df

def convert_df_to_3d_np_array(df: pd.DataFrame, numerical_type: np.object) -> np.array:
    # df_np_array = np.array(df, dtype= numerical_type)
    # df_np_array = df_np_array.reshape(df_np_array.shape[0],df_np_array.shape[1],1)
    df_np_array = np.zeros((df.shape[0],df.shape[1],df.iloc[0,0].shape[0]),dtype= numerical_type)
    for i in tqdm(range(df_np_array.shape[0])):
        for j in range(df_np_array.shape[1]):                    
                df_np_array[i,j] = df.iloc[i,j]
                
    return df_np_array

def shuffle_data_classification(input_ml_model_df: pd.DataFrame, input_series: pd.Series) -> tuple:
    joined_df = pd.concat([input_ml_model_df,input_series],axis=1)
    joined_df = joined_df.sample(frac=1).reset_index(drop=True)
    return joined_df.iloc[:,:-1], joined_df.iloc[:,-1]

def calculate_dtw_centroid(arrays: np.array):
    model = TimeSeriesKMedoids(n_clusters=1, metric="dtw")
    model.fit(arrays)
    return model.cluster_centers_[0][0]

def dtw_distance_to_centroid(arrays: np.array, centroid: np.array):
    distances = [dtw_distance(centroid, array) for array in arrays]
    return distances

def detect_and_remove_outliers_positive_centroid(df_activity: pd.DataFrame, series_classes: pd.Series, acivity_name: str) -> tuple:#(pd.DataFrame, pd.Series, np.array):
    np_array_activity = convert_df_to_3d_np_array( df_activity, np.single)

    
    centroid = calculate_dtw_centroid(np_array_activity)
    distances_list = dtw_distance_to_centroid(np_array_activity, centroid)

    auxilary_df = df_activity.copy()

    auxilary_df["distances_to_centroid"] = distances_list
    auxilary_df["classes"] = series_classes

    print("auxilary_df",auxilary_df)

    percentile_dist = np.percentile(distances_list, 75)
    
    df_activity_outlier_cleaned = auxilary_df[ auxilary_df["distances_to_centroid"] < percentile_dist]
    series_classes_outlier_cleaned = df_activity_outlier_cleaned["classes"]
    df_activity_outlier_cleaned.drop("classes",axis=1,inplace=True)
    df_activity_outlier_cleaned.drop("distances_to_centroid",axis=1,inplace=True)

    df_outliers = auxilary_df[ auxilary_df["distances_to_centroid"] >= percentile_dist]
    df_outliers.drop("classes",axis=1,inplace=True)
    df_outliers.drop("distances_to_centroid",axis=1,inplace=True)


    np_array_outliers = convert_df_to_3d_np_array( df_outliers, np.single)
    negative_centroid = calculate_dtw_centroid(np_array_outliers)
    negative_centroid_normalized = negative_centroid / auxilary_df.shape[0]

    print("negative_centroid_normalized", negative_centroid_normalized)
    
    return df_activity_outlier_cleaned , series_classes_outlier_cleaned , centroid #negative_centroid_normalized

def detect_and_remove_outliers_negative_centroid(df_activity: pd.DataFrame, series_classes: pd.Series, negative_centroid : np.array, acivity_name: str) -> tuple:#(pd.DataFrame, pd.Series):
    np_array_activity = convert_df_to_3d_np_array( df_activity, np.single)

    
    distances_list = dtw_distance_to_centroid(np_array_activity, negative_centroid)

    auxilary_df = df_activity.copy()

    print("auxilary_df", auxilary_df)
    print("df_activity", df_activity)

    auxilary_df["distances_to_centroid"] = distances_list
    auxilary_df["classes"] = series_classes

    print("auxilary_df", auxilary_df)
    print("df_activity", df_activity)

    percentile_dist = np.percentile(distances_list, 50)

    df_activity_outlier_cleaned = auxilary_df[ auxilary_df["distances_to_centroid"] > percentile_dist]
    series_classes_outlier_cleaned = df_activity_outlier_cleaned["classes"]
    df_activity_outlier_cleaned.drop("classes", axis=1, inplace=True)
    df_activity_outlier_cleaned.drop("distances_to_centroid", axis=1,inplace=True)
    
    return df_activity_outlier_cleaned , series_classes_outlier_cleaned

