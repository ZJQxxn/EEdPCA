# 数据集描述

实验中需要分析的数据集详细描述。

## Electricity Usage 数据

数据集链接：https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#

这个数据集记录了370个用户从2011年到2015年每一天内，每隔15分钟的用电量（单位为kW）。具体的数据集描述
可以参考上面的数据集链接。

我们选择2012到2014年三年内的数据。另外由于每个月的天数不同，对于每个月，我们只选择前10天和最后10天的数据。
最终处理后得到的数据大小为```(3, 12, 20, 96, 370)```，表示```(3年, 12个月, 20天, 96个时间段, 370个用户)```.
处理之后的数据存储在```./processed_data/ElectricityUsage.npy```。目前只是对数据进行了提取，使用时还需要进一步的 normalization
等操作。

#### 实验分析

 我们的数据中 ```n_sample = 3```，另外4个维度是 ```num_month=12, num_day=20, num_interval=96, num_client=370```。
需要：（1）分别分析4个维度的 component，（2）分析每个时间维度和 client 的 component，从不同的 timescale 来分析数据。


---------

