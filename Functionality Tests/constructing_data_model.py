import pandas as pd
import numpy as np
from tqdm import tqdm

from context import normalyzes_numerical_values_df, compensates_drift_with_PCA, reduces_dimensionality_by_PCA, House_Activitys_Learning_Base

def main():

    sync_df = pd.read_csv('../Generated Data/sync_df_without_piano.csv')#[:][377160:].reset_index(drop=True)
    #sync_df = pd.read_csv('./Generated Data/sync_df.csv')[:][:216103]

    #sync_df.describe().to_csv('./Generated Data/sync_df_describe.csv')

    sync_df["Time"] = pd.to_datetime(sync_df["Time"], format='%Y-%m-%d %H:%M:%S') 


    sync_df_normalazyde = normalyzes_numerical_values_df(sync_df)

    print("sync_df_normalazyde \n", sync_df_normalazyde, sync_df_normalazyde.shape)

    drift_conpensated_df = compensates_drift_with_PCA(sync_df_normalazyde)

    print("drift_conpensated_df \n", drift_conpensated_df, drift_conpensated_df.shape)

    sync_df_pca_reducted = reduces_dimensionality_by_PCA(drift_conpensated_df,1)


    #pd.DataFrame.to_csv(sync_df_pca_reducted,'./Generated Data/sync_df_pca_reducted.csv')

    activitys_df = pd.read_csv('../Activitys_time_schedule.csv') 
    ml_training_data_structure = House_Activitys_Learning_Base(10, sync_df_pca_reducted, activitys_df, 1, np.single)

    ml_training_data_structure.save_data_to_csv('../Generated Data')
    '''

    activitys_dict = ml_training_data_structure.data

    for key ,activ in activitys_dict.items():
        print("\n", key, "\n")
        for exec in activ.data:
            print(exec.data.size) 
    #'''
    train_input_ml_model_df, train_input_series, test_input_ml_model_df, test_input_series = ml_training_data_structure.prepare_data_input_classification(
        ngram_size = 50, 
        # days_for_train=["2022-11-14","2022-11-15","2022-11-16","2022-11-17","2022-11-18","2022-11-19","2022-11-20"], 
        # days_for_test=["2022-11-21","2022-11-22"])
        percentage_for_test = 20,
        percentage_for_train=70)
    print(train_input_ml_model_df.shape, train_input_series.shape, test_input_ml_model_df.shape, test_input_series.shape)

    np.save('../Generated Data/train_input_ml_model_df.npy',train_input_ml_model_df)
    np.save('../Generated Data/train_input_series.npy',train_input_series)
    np.save('../Generated Data/test_input_ml_model_df.npy',test_input_ml_model_df)
    np.save('../Generated Data/test_input_series.npy',test_input_series)

if __name__ == "__main__":
    main()
