import pandas as pd
from sync_lib import *
time_range = pd.date_range(start='2014-01-01 00:04:23', end='2014-01-01 00:06:23', freq="10S") + pd.Timedelta(seconds=1)
print(time_range)

df_test = pd.DataFrame({"col1":[1,2,3,4],"Time":['2014-01-01 00:04:23','2014-01-01 00:05:03','2014-01-01 00:05:43','2014-01-01 00:06:23']})
df_toma = pd.DataFrame({"col1":[1,2,3,4],"gimme":[89,3,45,32]})
#new_col = create_mv_avg_df(df_test,40,2)
#df_test = df_test.iloc[:new_col.shape[0],:] 
#df_test["newcol"] = new_col
print(df_toma)
for col in df_toma:
    df_toma[col] = np.zeros(df_toma.shape[0])
print(df_toma)
'''
initial_time_value = pd.to_datetime("2014-01-01 00:04:53")
print(initial_time_value.second % 20)
initial_time_value = initial_time_value.replace(second = (initial_time_value.second // 20)*20)
print(initial_time_value)

#'''
#print((34.512+ 34.233 +34.128 +34.464 +34.521 + 34.753)/6)
print((41.832 +
41.932 +
41.832 +
41.81 +
41.854 +
41.92 )/6)

df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', "é pau", 'foo'],
                    'value': [1, 2, 3, 4 , 5]})
df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo',"é pedra"],
                    'value': [5, 6, 7, 8, 4]})
df3 = df2.merge(df1,how="inner",on="value")

print(df1)

#df1.drop(index=[1,4],inplace=True)

print(df1)
print("\n\n\n")

df4,df5 = select_intersecting_rows(df1,df2,"value")

print(df4)
print("\n\n\n")
print(df5)

'''

def drop_nan_mod(df_mod1,df_mod2,df1_prefix,df2_prefix,col_in_common = "Time"):
    time_col_mod1 = df_mod1["Time"][:]
    time_col_mod2 = df_mod2["Time"][:]
    time_col_merge_mods = time_col_mod1.merge(time_col_mod2,how="inner",on=col_in_common)
    df_mod1_intersect = df_mod1.loc[(df_mod1['Time'].isin(time_col_merge_mods['Time']))]
    df_mod2_intersect = df_mod2.loc[(df_mod2['Time'].isin(time_col_merge_mods['Time']))]
    list_columns_df = list(df_mod1.columns)
    list_columns_df.remove(col_in_common)
    delete_flag = True
    for row in time_col_merge_mods.index[1:]:
        for i in range(5):
            if not (pd.isna(df_mod1_intersect.at[row + i,df_mod1_intersect.columns] and pd.isna(df_mod2_intersect.at[row + 1,df_mod2_intersect.columns]))):
                delete_flag = False
                break
        if delete_flag == True:
            for d in range(5):    
                df_mod1.drop(index=row + d,inplace=True)
                df_mod2.drop(index=row + d,inplace=True)
        delete_flag = True

    return 10
'''