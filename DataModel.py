import pandas as pd 
class DataModel:
    def __init__(self, path):
        self._path = path

    def getData(self) -> pd.DataFrame:
        print(self._path)
        data = pd.read_csv(self._path, names=['a','b','c','d','e','f','g','h'])
        #print(data)
        return data
