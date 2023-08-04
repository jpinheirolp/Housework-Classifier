import pandas as pd

class Execution_Activity():
    def __init__(self, execution_df: pd.DataFrame, begining_time: pd.Timestamp, end_time: pd.Timestamp, comment: str) -> None : 
        self.assert_argument_types(execution_df, begining_time, end_time)
        
        if not isinstance(comment, str):
            comment = "no comment"
        
        self.data = execution_df
        self.begining_time = begining_time
        self.end_time = end_time
        self.comment = comment

    def assert_argument_types(self, execution_df: pd.DataFrame, begining_time: pd.Timestamp, end_time: pd.Timestamp) -> None:
        assert isinstance(execution_df, pd.DataFrame), "execution_df must be a pandas DataFrame"
        assert isinstance(begining_time, pd.Timestamp),  "begining_time must be a pandas Timestamp"  
        assert isinstance(end_time, pd.Timestamp), "end_time must be a pandas Timestamp"

    def get_data(self) -> pd.DataFrame:
        return self.data
    
    def get_begining_time(self) -> pd.Timestamp:
        return self.begining_time
    
    def get_end_time(self) -> pd.Timestamp:
        return self.end_time
    
    def get_comment(self) -> str:
        return self.comment