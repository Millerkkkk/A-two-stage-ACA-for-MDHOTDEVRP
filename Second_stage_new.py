#  coding: UTF-8  #
"""
@filename: Second stage.py
@author: Yingkai
@time: 2024-02-15
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import function_new as func

df = pd.read_csv(
    r'C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\fig 3\c102_21.txt',
    sep=r'\s+')

print(df.head())

W = 650
Q = 40

# The value of c is written incorrectly in the paper
# c2 = 0.5
# c1 = 120
# c3 = 0.3
# c4 = 0.6
total_customers = 100

columns = ['x', 'y', 'demand', 'ReadyTime', 'DueDate', 'ServiceTime']
for c in columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# improved K-means clustering
# step 1 : the ratio of the total demand of all customers to EV capacity
total_demand = df['demand'].sum()
r = np.ceil(total_demand / W).astype(int)
clusters = {i: set() for i in range(r)}  # Initialize each cluster center

# Step 2: Randomly select the coordinates of r customers as the initial cluster centers.
customer_rows = df[df['Type'] == 'c']
initial_cluster_centers = customer_rows.sample(r)
# print(initial_cluster_centers)
initial_center_index = initial_cluster_centers.index.tolist()

depot_rows = df[df['Type'] == 'd']
depot_index = depot_rows.index.tolist()
cs_rows = df[df['Type'] == 'f']
cs_index = cs_rows.index.tolist()

# c102
clusters = {
    0: {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 96, 106, 109, 110, 112, 113, 114,
        115, 116, 117, 118, 119, 120, 121},
    1: {41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
        69, 70, 71, 72, 73},
    2: {74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101,
        102, 103, 104, 105, 107, 108, 111}}


# *************************************
# Second stage: EV distribution route planning and optimal charging decision

# Procedure 1: Variable initialization
N = 30  # Number of ants
maxIter = 200  # Maximum number of iterations
Iter = 1  # Current iteration
OptimalCost = np.inf  # Initialize the lowest total distribution cost
OptimalRoute = None  # Initialize the optimal distribution route
mu1, mu2 = 1, 2
rand0 = 0.3

pheromone_matrix = []
for cluster, customer in clusters.items():
    customer_list = list(customer)
    customer_list.extend(depot_index)
    customer_list.extend(cs_index)
    pheromone_matrix_1 = func.initialize_pheromone_matrix(len(customer_list), 1)
    pheromone_matrix.append(pheromone_matrix_1)

total_distance_matrix1 = func.total_distance_matrix(df)
total_distance_matrix_df = pd.DataFrame(total_distance_matrix1)
# print(total_distance_matrix_df)


while Iter <= maxIter:
    # print(pheromone_matrix)
    # Procedure 2: Initial route construction.
    # step 1
    ant_nodes = {i: [] for i in clusters}
    for cluster, customer in clusters.items():
        num_customers = len(customer)
        num_ant = round((num_customers / total_customers) * N)
        ant_nodes[cluster] = set(random.sample(list(customer), num_ant))
    print('ant_nodes', ant_nodes)

    # Generate a path for each ant
    FinalRoutes = {i: {} for i in range(r)}
    AntRoutes = {i: {} for i in range(r)}

    for cluster, customer in clusters.items():
        # Example Initialize the node index mapping
        node_to_index = {node: index for index, node in enumerate(customer)}
        index_to_node = {index: node for index, node in enumerate(customer)}

        # step 2
        customer_list = list(customer)
        ant_list = list(ant_nodes[cluster])
        distance_matrix = func.initialize_distance_matrix(customer_list, df)
        heuristic_matrix = func.initialize_heuristic_matrix(distance_matrix)
        print('customer', customer)

        for ant_1 in ant_nodes[cluster]:
            # print(ant_1)
            current_index = node_to_index[ant_1]
            # print(current_index)
            # Initializes unvisited collections and taboos
            unvisited = set(customer)
            unvisited.remove(ant_1)

            tabu = [ant_1]
            # print(unvisited)
            # print(tabu)

            # 进行路径选择
            while unvisited:
                # Step 3 Computed transition probability
                # print(unvisited)
                probability, numerator = func.transition(pheromone_matrix[cluster], heuristic_matrix, unvisited,
                                                         current_index)
                # print('probability', probability)
                # Select the next node
                next_node = func.select_next_node(probability, numerator, unvisited, 1)
                next_index = node_to_index[next_node]
                # print('next_node', next_node)
                # print('next_index', next_index)
                # Update the current node and tabu table
                current_index = next_index
                tabu.append(next_node)
                unvisited.remove(next_node)
            # print('tabu', tabu)

            AntRoutes[cluster][ant_1] = tabu
    print('AntRoutes', AntRoutes)

    # Procedure 3: Depot allocation and charging station insertion.
    # Procedure 4: Calculation of the total distribution cost for each ant
    total_ant_routes = {}
    total_ant_costs = {}
    dispatch_cost1 = travel_cost1 = service_cost1 = charging_cost1 = 0
    for cluster, ants in AntRoutes.items():
        allocated_depot_route = func.allocate_depots(df, ants)  # 分配depot
        # print(allocated_depot_route)
        total_costs = {}
        finalRoutes = {}

        for ant, route in allocated_depot_route.items():
            add_cs, travel_time, charging_time = func.insert_cs(df, route, total_distance_matrix_df, 0)  # 插入CS
            finalRoutes[ant] = add_cs
            # dispatch_cost = func.total_cost(df, add_cs, travel_time, charging_time)
            # total_costs[ant] = dispatch_cost

            dispatch_cost, travel_cost, service_cost, charging_cost = func.total_cost(df, add_cs, travel_time,
                                                                                      charging_time)
            total_costs[ant] = dispatch_cost + travel_cost + service_cost + charging_cost
            print('EC, VC, SC, CC', dispatch_cost, travel_cost, service_cost, charging_cost)
        # dispatch_cost1 += dispatch_cost
        # travel_cost1 += travel_cost
        # service_cost1 += service_cost
        # charging_cost1 += charging_cost

        total_ant_routes[cluster] = finalRoutes
        total_ant_costs[cluster] = total_costs

    print('total_ant_costs', total_ant_costs)
    print('total_ant_routes', total_ant_routes)

    # select the optimal path
    min_cost_ants = {}
    min_cost_routes = {}
    total_min_cost = 0
    for cluster, ants in total_ant_costs.items():
        # Find the ant with the minimum cost in the current cluster
        min_ant = min(ants, key=ants.get)
        min_cost = ants[min_ant]
        total_min_cost += min_cost

        # Retrieve the corresponding route from total_ant_routes
        min_route = total_ant_routes[cluster][min_ant]

        # Store the results in the dictionaries
        min_cost_ants[cluster] = (min_ant, min_cost)
        min_cost_routes[cluster] = min_route

    print('********** ', Iter, ' **********')
    print('min_cost_ants:', min_cost_ants)
    print('min_cost_routes:', min_cost_routes)
    print('total_min_cost:', total_min_cost)

    # Procedure 5: Pheromone update
    # update the pheromone matrix
    iter_pheromone_matrix = []
    for cluster, customer in clusters.items():
        # 要将每个clustering中加入depot和CS
        customer_list = list(customer)
        customer_list.extend(depot_index)
        customer_list.extend(cs_index)

        iter_pheromone_matrix1 = func.update_pheromone(pheromone_matrix[cluster], customer_list,
                                                       total_ant_routes[cluster], total_ant_costs[cluster],
                                                       min_cost_ants[cluster], rho=0.2, f=10, epsilon=0.1, t_min=0.05,
                                                       t_max=2)
        iter_pheromone_matrix.append(iter_pheromone_matrix1)
        # print(iter_pheromone_matrix)
    pheromone_matrix = iter_pheromone_matrix
    Iter += 1
else:
    # Output the final optimized result
    print('Final Optimized Result:')
    print('Total_Minimum_Cost = ', total_min_cost)
    print('Minimum_Cost_Routes = ', min_cost_routes)
