import pandas as pd
from sync_lib import *
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

#'''

libelium_mod1_df = pd.read_table("./Libelium new/mod1.txt",sep=';')
libelium_mod2_df = pd.read_table("./Libelium new/mod2.txt",sep=';')

# libelium_mod1_df = libelium_mod1_df.dropna()
# libelium_mod2_df = libelium_mod2_df.dropna()

print(libelium_mod1_df.columns)
print(libelium_mod2_df.columns) 

#libelium_mod2_df.drop("Time",axis=1, inplace= True)


libelium_mod3_df = libelium_mod2_df.loc[:][:100]
libelium_mod4_df = libelium_mod1_df.loc[:][:100]

#libelium_mod3_df["Time"] = libelium_mod3_df["Time"].astype("datetime64[s]",copy=False)

libelium_mod3_df, libelium_mod4_df = remove_nan_dfs(libelium_mod3_df, libelium_mod4_df)


#libelium_mod3_df = fills_empty_values(libelium_mod3_df.copy())

libelium_mod2_freq20_df = create_mv_avg_df(libelium_mod3_df,20)
libelium_mod1_freq20_df = create_mv_avg_df(libelium_mod4_df,20)




print(libelium_mod3_df)
print(libelium_mod4_df)



print(libelium_mod2_freq20_df.columns)

# libelium_concat_freq20_df = merge_ajusting_col_names(libelium_mod1_freq20_df,libelium_mod2_freq20_df,"mod1","mod2")
# pd.DataFrame.to_csv(libelium_concat_freq20_df,'./Generated Data/libelium_concat_freq20.csv')
# pd.DataFrame.to_csv(libelium_mod2_freq20_df,'./Generated Data/libelium_mod2_freq20.csv')
# pd.DataFrame.to_csv(libelium_mod1_freq20_df,'./Generated Data/libelium_mod1_freq20.csv')



#libeliums_concat = pd.concat( [libelium_mod1_df , libelium_mod2_df], axis=1)

#libeliums_concat.drop(libeliums_concat.loc[1761456:].index,axis=0,inplace=True)

#pd.DataFrame.to_csv(libeliums_concat,'./Generated Data/libelium_concat.csv')



#'''

# finding first ocurrance os different value in Time column
'''
for l in range(617035):
    if pods_85_df["Time"].iloc[l] != pods_86_df["Time"].iloc[l]:
        print(l)
        break
'''
