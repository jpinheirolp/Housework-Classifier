import pandas as pd
from sync_lib import *

# Reading the piano sensor files and concatenating them into piano_concat.csv 
'''
piano_thick_df = pd.read_csv("./Piano/IMT_Thick.csv",sep=';')
piano_thin_df = pd.read_csv("./Piano/IMT_Thin.csv",sep=';')
piano_pico_df = pd.read_csv("./Piano/IMT_PICO.csv",sep=';')


piano_thin_df.drop([77761],axis=0, inplace= True) #



piano_thin_df.index = pd.RangeIndex(start=0, stop=617034, step=1)


print(piano_thick_df.shape)
print(piano_thin_df.shape)
print(piano_pico_df.shape)

piano_thin_df.drop("Time",axis=1, inplace= True) 
piano_pico_df.drop("Time",axis=1, inplace= True)

columns_piano_thick_renamed = []
for col in piano_thick_df.columns:
    columns_piano_thick_renamed.append("piano_thick_" + col)  
piano_thick_df.columns = columns_piano_thick_renamed

columns_piano_thin_renamed = []
for col in piano_thin_df.columns:
    columns_piano_thin_renamed.append("piano_thin_" + col)  
piano_thin_df.columns = columns_piano_thin_renamed

columns_piano_pico_renamed = []
for col in piano_pico_df.columns:
    columns_piano_pico_renamed.append("piano_pico_" + col)  
piano_pico_df.columns = columns_piano_pico_renamed

piano_concat = pd.concat([piano_thick_df, piano_thin_df, piano_pico_df],axis=1)



pd.DataFrame.to_csv(piano_concat,'./Generated Data/piano_concat.csv')

#'''

# Reading the pod sensor files and concatenating them into pod_concat.csv

'''

pods_85_df = pd.read_csv("./Pods/POD 200085.csv",sep=';')
pods_86_df = pd.read_csv("./Pods/POD 200086.csv",sep=';')
pods_88_df = pd.read_csv("./Pods/POD 200088.csv",sep=';')

pods_85_df.drop(245520,axis=0, inplace=True)
pods_85_df.index = pd.RangeIndex(start=0, stop=617034, step=1)

pods_86_df.drop(617034,axis=0, inplace= True)


pods_85_df.drop(["co2", "ec3", "ec4", "p1"],axis=1, inplace= True)
pods_86_df.drop(["Time", "co2", "ec3", "ec4", "p1"],axis=1, inplace= True)
pods_88_df.drop(["Time", "co2", "ec3", "ec4", "p1"],axis=1, inplace= True)

# renaming columns
columns_pod_85_renamed = []
for col in pods_85_df.columns:
    columns_pod_85_renamed.append("pod_85_" + col)  
pods_85_df.columns = columns_pod_85_renamed

columns_pod_86_renamed = []
for col in pods_86_df.columns:
    columns_pod_86_renamed.append("pod_86_" + col)  
pods_86_df.columns = columns_pod_86_renamed

columns_pod_88_renamed = []
for col in pods_88_df.columns:
    columns_pod_88_renamed.append("pod_88_" + col)  
pods_88_df.columns = columns_pod_88_renamed

print(pods_85_df.columns)
print(pods_86_df.columns)
print(pods_88_df.columns)

pods_concat = pd.concat([pods_85_df,pods_86_df,pods_88_df],axis=1)

pd.DataFrame.to_csv(pods_concat,'./Generated Data/pod_concat.csv')

# '''

# Reading the libellium sensor files, changing there frequence to 10 with the create_mv_avg_df func and concatenating them into libelium_concat_freq10.csv

'''

libelium_mod1_df = pd.read_table("./Libelium new/mod1.txt",sep=';')#, parse_dates=["Time"])
libelium_mod2_df = pd.read_table("./Libelium new/mod2.txt",sep=';')#, parse_dates=["Time"])

# libelium_mod1_df = libelium_mod1_df.dropna()
# libelium_mod2_df = libelium_mod2_df.dropna()


#libelium_mod2_df.drop("Time",axis=1, inplace= True)

libelium_mod3_df = libelium_mod2_df.loc[:][:]#.reset_index(drop=True)
libelium_mod4_df = libelium_mod1_df.loc[:][:]#.reset_index(drop=True)

libelium_mod3_df["Time"] = pd.to_datetime(libelium_mod3_df["Time"], format='%d/%m/%Y %H:%M:%S')
libelium_mod4_df["Time"] = pd.to_datetime(libelium_mod4_df["Time"], format='%d/%m/%Y %H:%M:%S')

#libelium_mod3_df["Time"] = libelium_mod3_df["Time"].astype("datetime64[s]",copy=False)

#libelium_mod3_df, libelium_mod4_df = remove_nan_dfs(libelium_mod3_df, libelium_mod4_df)

libelium_mod3_df = fills_empty_values(libelium_mod3_df)
libelium_mod4_df = fills_empty_values(libelium_mod4_df)

print("fo00i\n\n\n\n")

print(libelium_mod3_df)
print(libelium_mod4_df)


#libelium_mod3_df = fills_empty_values(libelium_mod3_df.copy())

libelium_mod2_freq10_df = create_mv_avg_df(libelium_mod3_df,10)
libelium_mod1_freq10_df = create_mv_avg_df(libelium_mod4_df,10)





libelium_concat_freq10_df = merge_ajusting_col_names(libelium_mod1_freq10_df,libelium_mod2_freq10_df,"mod1","mod2")
print(libelium_concat_freq10_df.shape)
pd.DataFrame.to_csv(libelium_concat_freq10_df,'./Generated Data/libelium_concat_freq10.csv')
pd.DataFrame.to_csv(libelium_mod2_freq10_df,'./Generated Data/libelium_mod2_freq10.csv')
pd.DataFrame.to_csv(libelium_mod1_freq10_df,'./Generated Data/libelium_mod1_freq10.csv')



#libeliums_concat = pd.concat( [libelium_mod1_df , libelium_mod2_df], axis=1)

#libeliums_concat.drop(libeliums_concat.loc[1761456:].index,axis=0,inplace=True)

#pd.DataFrame.to_csv(libeliums_concat,'./Generated Data/libelium_concat.csv')



#'''

#Not used in the final csv file: I was just changing the frequency of pod and piano to 20 

'''
pod_concat_df = pd.read_csv('./Generated Data/pod_concat.csv')
piano_concat_df = pd.read_csv('./Generated Data/piano_concat.csv')
libellium_conct_freq20 = pd.read_csv('./Generated Data/libelium_concat_freq20.csv')

pod_concat_df.rename(columns={"pod_85_Time": "Time"}, inplace=True)
piano_concat_df.rename(columns={"piano_thick_Time": "Time"}, inplace=True)

pod_concat_df["Time"] = pd.to_datetime(pod_concat_df["Time"])
piano_concat_df["Time"] = pd.to_datetime(piano_concat_df["Time"])
libellium_conct_freq20["Time_Interval"] = pd.to_datetime(libellium_conct_freq20["Time_Interval"])

piano_concat_df_freq20 = create_mv_avg_df(piano_concat_df,20)
pod_concat_df_freq20 = create_mv_avg_df(pod_concat_df,20)

pd.DataFrame.to_csv(pod_concat_df_freq20,'./Generated Data/pod_concat_df_freq20.csv')
pd.DataFrame.to_csv(piano_concat_df_freq20,'./Generated Data/piano_concat_df_freq20.csv')

#'''

# merging all pod_concat.csv, piano_concat.csv, libelium_concat_freq10.csv into sync.csv and ajusting column names

'''
pod_concat = pd.read_csv('./Generated Data/pod_concat.csv')#[:][:100000]
piano_concat = pd.read_csv('./Generated Data/piano_concat.csv')#[:][:100000]
libellium_concat_freq10 = pd.read_csv('./Generated Data/libelium_concat_freq10.csv')#[:][:100000]

libellium_concat_freq10.rename(columns = {"Time_Interval":"Time"},inplace=True)
libellium_concat_freq10["Time"] = pd.to_datetime(libellium_concat_freq10["Time"], format='%Y-%m-%d %H:%M:%S') 

pod_concat.rename(columns={"pod_85_Time": "Time"}, inplace=True)
piano_concat.rename(columns={"piano_thick_Time": "Time"}, inplace=True)




pod_concat["Time"] = pd.to_datetime(pod_concat["Time"], infer_datetime_format=True)
piano_concat["Time"] = pd.to_datetime(piano_concat["Time"], infer_datetime_format=True)


print(pod_concat.columns)
print(piano_concat.columns)
print(libellium_concat_freq10.columns)

pod_piano_concat_df = pod_concat.merge(piano_concat,how="inner",on="Time")

pd.DataFrame.to_csv(pod_piano_concat_df,'./Generated Data/pod_piano_concat.csv')


pod_piano_set = set(pod_piano_concat_df["Time"])
intersection_set = pod_piano_set.intersection(set(libellium_concat_freq10["Time"]))
print(len(pod_piano_set - intersection_set))

sync_df = pod_piano_concat_df.merge(libellium_concat_freq10,how="inner",on="Time")

sync_df.drop(["Unnamed: 0_x","Unnamed: 0_y"],axis=1,inplace=True)

pd.DataFrame.to_csv(sync_df,'./Generated Data/sync_df.csv')

#  ''' 

#removind strange columns that appeared
'''

sync_df = pd.read_csv('./Generated Data/sync_df.csv')
print(sync_df.columns,sync_df.shape)
sync_df.drop(["Unnamed: 0","Unnamed: 0.1"],axis=1,inplace=True)
pd.DataFrame.to_csv(sync_df,'./Generated Data/sync_df.csv')

'''