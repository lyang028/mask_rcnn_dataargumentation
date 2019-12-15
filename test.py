import csv

def save_data(data, path):

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame(data)

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(path, index=False,header=False)