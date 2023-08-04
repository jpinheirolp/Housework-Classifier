import numpy as np
import pandas as pd
from context import Execution_Activity

def generate_random_time_stamp(random_year, random_month, random_day, random_hour_gap = 0):
    random_year, random_month, random_day

    random_hour = np.random.randint(0, 11)
    random_minute = np.random.randint(0, 59)
    random_second = np.random.randint(0, 59)

    return pd.Timestamp(f"{random_year}-{random_month}-{random_day} {random_hour + random_hour_gap}:{random_minute}:{random_second}")

def generate_execution_mock():
    random_year = np.random.randint(2020, 2030)
    random_month = np.random.randint(1, 12)
    random_day = np.random.randint(1, 28)
    random_execution_hour_gap = np.random.randint(1, 12)

    begining_time = generate_random_time_stamp(random_year, random_month, random_day)
    end_time = generate_random_time_stamp(random_year, random_month, random_day, random_execution_hour_gap)

    time_column_mock = pd.date_range(start= begining_time, end= end_time, freq= "10S")
    eigenvalue_column_mock_1 = np.random.random(size = len(time_column_mock))
    eigenvalue_column_mock_2 = np.random.random(size = len(time_column_mock))

    data = pd.DataFrame({
        "Time": time_column_mock, 
        "Eigenvalue_1": eigenvalue_column_mock_1,
        "Eigenvalue_2": eigenvalue_column_mock_2 
        })
    
    comment = 'This is a comment'
    execution_activity = Execution_Activity(data, begining_time, end_time, comment)

    return execution_activity, data, begining_time, end_time