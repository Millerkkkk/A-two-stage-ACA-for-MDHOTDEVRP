#  coding: UTF-8  #
"""
@filename: Clustering.py
@author: Yingkai
@time: 2024-02-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import function_new as func


W = 650
Q = 40

c2 = 0.5
c1 = 120
c3 = 0.3
c4 = 0.6
total_customers = 100


df = pd.read_csv(r'C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\fig 3\c102_21.txt', sep=r'\s+')
print(df.head())


columns = ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']
for c in columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')


# improved K-means clustering
# step 1 : the ratio of the total demand of all customers to EV capacity
total_demand = df['demand'].sum()
r = np.ceil(total_demand / W).astype(int)
clusters = {i : set() for i in range(r)}    # Initialize each cluster center

# Step 2: Randomly select the coordinates of r customers as the initial cluster centers.
customer_rows = df[df['Type'] == 'c']
initial_cluster_centers = customer_rows.sample(r)
# print(initial_cluster_centers)
initial_center_index = initial_cluster_centers.index.tolist()

depot_rows = df[df['Type'] == 'd']
depot_index = depot_rows.index.tolist()
cs_rows = df[df['Type'] == 'f']
cs_index = cs_rows.index.tolist()


# Step 3: Set Cap to represent the capacity of each cluster, where Cap equals the EV capacity. add all unassigned customers to a set called unallocate.
EV_capacity = W
cluster_capacities = {i: EV_capacity for i in range(r)}


unallocated = set(customer_rows.index)
# print(unallocated)


while True:
    while unallocated:
        customer_idx = df.loc[list(unallocated), 'demand'].idxmax()
        customer_demand = df.loc[customer_idx, 'demand']
        customer_location = df.loc[customer_idx, ['x', 'y']].values
        # print('customer_location', customer_location)

        # Select customer with the largest demand, calculate distance to cluster centers
        dis1 = []
        for i in range(r):
            center_location = initial_cluster_centers.loc[initial_center_index[i], ['x', 'y']].values
            dis2 = func.distance(customer_location, center_location)
            dis1.append(dis2)

        # Determine the nearest cluster, Assign customer to cluster if capacity allows, else assign to second nearest
        sorted_indices = np.argsort(dis1)
        for i in range(len(sorted_indices)):
            nearest_cluster = sorted_indices[i]
            if cluster_capacities[nearest_cluster] >= customer_demand:
                clusters[nearest_cluster].add(customer_idx)
                cluster_capacities[nearest_cluster] -= customer_demand
                unallocated.remove(customer_idx)
                break

    new_cluster_centers = func.recalculate_centroids(clusters, df)
    # print('11111', new_cluster_centers)

    threshold = 10
    centers_changed = False
    for i in range(r):
        old_center = initial_cluster_centers.loc[initial_center_index[i], ['x', 'y']].values
        new_center = new_cluster_centers[i]
        if np.linalg.norm(new_center - old_center) > threshold:
            centers_changed = True
            # print('new-old distance: ', np.linalg.norm(new_center - old_center))
            break

    if centers_changed:
        initial_cluster_centers = pd.DataFrame.from_dict(new_cluster_centers, orient='index')
        initial_cluster_centers.columns = ['x', 'y']  # 设置列名
        initial_center_index = initial_cluster_centers.index.tolist()  # 更新索引列表
        unallocated = set(customer_rows.index)
        clusters = {i: set() for i in range(r)}
        cluster_capacities = {i: EV_capacity for i in range(r)}
        # print(clusters)
    else:
        print("Cluster assignments:", clusters)
        break


def plot_clusters(clusters2, df2):
    plt.figure(figsize=(10, 8))

    # 遍历每个群集及其客户索引
    for cluster_idx, customer_indices in clusters.items():
        # 将集合转换为列表
        customer_indices_list = list(customer_indices)
        cluster_points = df.loc[customer_indices_list, ['x', 'y']]
        plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {cluster_idx}')

    # # 绘制充电站节点
    # cs_points = df[df['Type'] == 'f']
    # plt.scatter(cs_points['x'], cs_points['y'], color='black', marker='^', label='Charging Station')
    #
    # # 绘制仓库节点
    # depot_points = df[df['Type'] == 'd']
    # plt.scatter(depot_points['x'], depot_points['y'], color='black', marker='s', label='Depot')

    plt.title('Cluster Assignments')
    plt.xlabel('X-label')
    plt.ylabel('Y-label')
    plt.legend()
    plt.show()

# 调用函数绘制聚类图
plot_clusters(clusters, df)











