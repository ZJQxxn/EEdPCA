import numpy as np
import pandas as pd
import copy


def elecUse():
    # 读取数据；预处理
    data = pd.read_csv("./data/ElectricityUsage/LD2011_2014.txt", sep=";")
    data = data.rename({"Unnamed: 0": "date"})
    data["year"] = data["Unnamed: 0"].apply(lambda x: x.split(" ")[0].split("-")[0])
    data["month"] = data["Unnamed: 0"].apply(lambda x: x.split(" ")[0].split("-")[1])
    data["day"] = data["Unnamed: 0"].apply(lambda x: x.split(" ")[0].split("-")[2])
    data["time"] = data["Unnamed: 0"].apply(lambda x: x.split(" ")[1])

    # 划分数据
    all_data = []
    group_by_year = data.groupby("year")
    print("The number of years : ", len(group_by_year))
    for year, each_year in group_by_year:
        # 只使用 2012~2014 年的数据
        if year == "2011" or year == "2015":
            continue
        year_data = []
        print("=" * 50)
        print(each_year.year.values[0])
        group_by_month = each_year.groupby("month")
        print("The number of months : ", len(group_by_month))
        for month, each_month in group_by_month:
            month_data = []
            print("-" * 50)
            print(each_month.month.values[0])
            group_by_day = each_month.groupby("day")
            day_list = sorted(group_by_day.day.apply(lambda x: x.values[0]).unique())
            print("The number of days : ", len(group_by_day))
            for day in day_list[:10] + day_list[-10:]:
                each_day = group_by_day.get_group(day).filter(regex="MT", axis=1)
                # 数据转换为 float
                for col in each_day.columns.values:
                    each_day[col] = pd.to_numeric(
                        each_day[col].apply(lambda x: x.replace(",", ".") if isinstance(x, str) else float(x)),
                        downcast="float")
                month_data.append(copy.deepcopy(each_day.values))
            year_data.append(copy.deepcopy(month_data))
        all_data.append(copy.deepcopy(year_data))
    # 保存数据
    all_data = np.array(all_data)
    print("\n All data shape : ", all_data.shape)
    np.save("./processed_data/ElectricityUsage.npy", all_data)


if __name__ == '__main__':
    elecUse() # 提取 Electricity Usage 数据