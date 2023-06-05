import csv
import os

def read_csv_file(file_path):
    data = []

    with open(file_path, 'r', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)

    return data

# 获取当前工作目录
current_directory = os.getcwd()

# 拼接CSV文件路径
csv_file_path = os.path.join(current_directory, 'remote_vehicles.csv')

# 读取CSV文件内容并存入内存
csv_data = read_csv_file(csv_file_path)

float_csv_data = []

for row in csv_data:
    float_row = []
    for value in row:
        if value != '':
            float_value = float(value)
            float_row.append(float_value)
        else:
            float_row.append(None)  # 或者根据需要添加适当的处理方式
    float_csv_data.append(float_row)

# # 打印转换后的浮点数数据
# for row in float_csv_data:
#     print(row)



# # 打印读取的数据
# for row in csv_data:
#     print(row)
#
# crash_area = csv_data[0]
#
# print(crash_area)
#
# first_field = csv_data[0][0]
#
# print(first_field)
# print(type(first_field))

