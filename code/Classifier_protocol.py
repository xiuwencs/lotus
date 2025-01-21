import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_percentage(line):
    """
        计算可打印字符占比
        :param line:预处理后的每一行消息
        :return: 可打印字符占比
        """
    count_valid = 0
    count_total = 0
    for i in range(0, len(line) - 1, 2):
        pair = line[i:i + 2]
        if 0x41 <= int(pair, 16) <= 0x5a or 0x61 <= int(pair, 16) <= 0x7a:
            count_valid += 1
        count_total += 1
    if count_total > 0:
        return (count_valid / count_total) * 100
    else:
        return 0


data = []
file_path = "mixed_protocol.txt"  #文本协议和二进制协议混合协议
with open(file_path, 'r') as file:
    line_number = 1
    for line in file:
        line = line.strip()
        percentage = calculate_percentage(line)
        data.append([percentage])
        # print(f"{line_number}:{percentage:.2f}%")
        line_number += 1

data = np.array(data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(data_scaled)

cluster1_data = []
cluster2_data = []
for i in range(len(clusters)):
    if clusters[i] == 0:
        cluster1_data.append(f"{i + 1}")
    else:
        cluster2_data.append(f"{i + 1}")
print("------In the mixed file------")
print("binary protocol:")
for i in range(0, len(cluster1_data), 40):
    print(",".join(cluster1_data[i:i + 40]))
print("text protocol:")
for i in range(0, len(cluster2_data), 40):
    print(",".join(cluster2_data[i:i + 40]))

cluster1_mean = np.mean(data[clusters == 0])
cluster2_mean = np.mean(data[clusters == 1])
boundary_point = (cluster1_mean + cluster2_mean) / 2
print(f"boundary:{boundary_point}%\n")

plt.figure()
plt.scatter(cluster1_data, [0] * len(cluster1_data), color='red', label='cluster 1')
plt.scatter(cluster2_data, [0] * len(cluster2_data), color='blue', label='cluster 2')

for i, data_point in enumerate(cluster1_data):
    plt.text(data_point, 0, str(i + 1))

for i, data_point in enumerate(cluster2_data):
    plt.text(data_point, 0, str(i + 6))


def check_file(file_path, boundary_point):
    """
        判断协议类型
        :param file_path:文件路径
        :param boundary_point:文本协议与二进制协议可打印字符占比的边界值
        :return: 协议类型
        """
    flag = 0
    total = 0
    count = 0
    with open(file_path, 'r') as file1:
        for line in file1:
            line = line.strip()
            if line:
                count += 1
                ratio = calculate_percentage(line)
                total += ratio
        avq_ratio = total / count
        print("avg_ratio:", avq_ratio)
        if avq_ratio < boundary_point:
            flag = 1
        else:
            flag = 0
    return flag


file_path1 = 'http.txt' #实验协议
flag = check_file(file_path1, boundary_point)
print("------The type of input protocol------ ")
if flag == 0:
    print("textual protocol")
else:
    print("binary protocol")
