#  coding: UTF-8  #
"""
@filename: function.py
@author: Yingkai
@time: 2024-02-15
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import math


# First stage
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def recalculate_centroids(clusters1, df1):
    new_centers = {}
    for key, value in clusters1.items():
        if value:
            list1 = list(value)
            all_customers = df1.loc[list1, ['x', 'y']]
            new_center = all_customers.mean().values
            new_centers[key] = new_center
    return new_centers


# Second stage
def initialize_distance_matrix(df1, df2):
    num_nodes = len(df1)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row1, row2 = df1[i], df1[j]
                p1 = df2.iloc[row1][['x', 'y']].values
                p2 = df2.iloc[row2][['x', 'y']].values
                distance_matrix[i][j] = distance(p1, p2)
    return distance_matrix


def total_distance_matrix(df1):
    num_nodes = len(df1)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # 不计算节点到自己的距离
                # 直接从DataFrame中获取坐标
                p1 = df1.iloc[i][['x', 'y']].values
                p2 = df1.iloc[j][['x', 'y']].values
                # 计算两点之间的欧几里得距离
                distance_matrix[i][j] = np.linalg.norm(p1 - p2)
    return distance_matrix


# Initialize the pheromone matrix
def initialize_pheromone_matrix(num_nodes, initial_pheromone):
    return np.full((num_nodes, num_nodes), initial_pheromone)


# Initialize the heuristic information matrix
def initialize_heuristic_matrix(distance_matrix):
    heuristic_matrix = np.where(distance_matrix != 0, 1.0 / distance_matrix, 0)
    return heuristic_matrix


# Calculate transition probability
def transition(pheromones, heuristic, unvisited, current_node, mu1 = 1, mu2 = 2):
    # Create a mapping from node number to matrix index
    node_to_index1 = {node: index for index, node in enumerate(unvisited)}
    probabilities = {}
    # Transform the index using a map, and then calculate the denominator of the probability
    numerator = {j: pheromones[current_node][node_to_index1[j]] ** mu1 *
                        heuristic[current_node][node_to_index1[j]] ** mu2
                     for j in unvisited}
    denominator = sum(numerator.values())

    for j in unvisited:
        probabilities[j] = (numerator[j] / denominator)
    return probabilities, numerator


# Select the next node
def select_next_node(probabilities, numerator, unvisited, rand0):
    rand = random.random()
    if rand <= rand0:
        # Select the node with the highest probability
        next_node = max(unvisited, key=lambda x: numerator.get(x, 0))
    else:
        # Filter out nodes with non-finite probabilities
        finite_probs = {node: prob for node, prob in probabilities.items() if math.isfinite(prob) and node in unvisited}
        print('finite_probs', finite_probs)
        # If all probabilities are non-finite, randomly select from unvisited nodes
        if not finite_probs:
            next_node = random.choice(list(unvisited))
            print('not finite_probs')
        else:
            # Make a weighted choice from nodes with finite probabilities
            nodes, probs = zip(*finite_probs.items())
            next_node = random.choices(nodes, weights=probs, k=1)[0]
    return next_node


# Calculate the speed of the tram at each node
def EV_speed(departure_time, travel_times, congestion_times):
    speeds = {}
    for node, travel_time in travel_times.items():
        # Calculate the time to reach each node
        arrival_time = departure_time + timedelta(minutes=travel_time)
        speed = 60

        # Determine whether the arrival time is within the congestion period
        for start, end in congestion_times:
            if start <= arrival_time <= end:
                speed = 30
                break

        speeds[node] = speed

    return speeds


def allocate_depots(dataframe, InitialRoutes):
    depot_rows = dataframe[dataframe['Type'] == 'd']
    depot_index = set(depot_rows.index)
    CS_rows = dataframe[dataframe['Type'] == 'f']
    CS_index = set(CS_rows.index)

    FinalRoutes = {}
    for cluster, route in InitialRoutes.items():
        # Step 2: Depot allocation
        # Assuming depot_index contains the indices of depots, and you have a dataframe df containing all locations
        start_customer_location = dataframe.loc[route[0], ['x', 'y']].values
        end_customer_location = dataframe.loc[route[-1], ['x', 'y']].values
        # print(route[0])
        # print(f"起始客户位置: {start_customer_location}, 结束客户位置: {end_customer_location}")

        Mi = min(depot_index, key=lambda depot_idx: distance(start_customer_location, dataframe.loc[depot_idx, ['x', 'y']].values))
        Mj = min(depot_index, key=lambda depot_idx: distance(end_customer_location, dataframe.loc[depot_idx, ['x', 'y']].values))
        # print('Mi', Mi)
        # print('Mj', Mj)

        # Insert the depots at the start and end of the route
        FinalRoute = [Mi] + route + [Mj]
        FinalRoutes[cluster] = FinalRoute
    return FinalRoutes


def get_distances_from_matrix(distance_matrix, nodes_list):
    distances = {}
    for key, indices in nodes_list.items():
        path_distances = []
        for i in range(len(indices)-1):
            start_node = indices[i]
            end_node = indices[i+1]
            path_distances.append(distance_matrix[start_node][end_node])
        distances[key] = path_distances
    return distances


def calculate_energy_consumption(phi_motor, phi_battery, g, theta_ij, C_r, L, u_ijk, R, A, rho, v_ijk_R, t_ijk_R):
    # The rolling and air resistance components of energy consumption
    resistance_energy = (g * np.sin(theta_ij) + C_r * g * np.cos(theta_ij)) * (L + u_ijk) / 3600
    # The aerodynamic drag component of energy consumption
    aerodynamic_drag_energy = (R * A * rho * v_ijk_R ** 2) / 76140
    # Total energy consumption for the EV
    e_ijk_R = phi_motor * phi_battery * (resistance_energy + aerodynamic_drag_energy) * v_ijk_R * t_ijk_R
    return e_ijk_R


# Add the charging time calculation based on the formulas provided
def calculate_charging_time(q_ik):
    # Convert minutes to hours for consistency
    return (60 * q_ik) / (0.9 * 60)


def energy_time(distance1, departure_time1, u_ijk):
    max_periods = 24  # Maximum number of time periods
    period_length = 1  # Time period length
    speed = 0
    total_energy = 0
    travel_time = 0
    remaining_distance = distance1

    # print('distance1, departure_time1, u_ijk', distance1, departure_time1, u_ijk)
    while departure_time1 <= max_periods:
        period = int(departure_time1)
    # for period in range(int(departure_time1), max_periods):
        # Calculates the start and end time of the current time range
        period_start = period * period_length
        period_end = (period + 1) * period_length

        # Set the speed according to the time period
        if (period_start >= 0 and period_end <= 2) or (period_start >= 10 and period_end <= 12):
            speed = 30
        else:
            speed = 60

        # Calculates the travel time in the current time period
        t_ijk_R = min(period_end - departure_time1, remaining_distance / speed)
        # t_ijk_R = min(period_end - departure_time1, Fraction(remaining_distance, speed))

        # Calculate the energy consumption in the current time period
        energy_this_period = calculate_energy_consumption(1.184692, 1.112434, 9.8, 0,
                                                          0.012, 3000, u_ijk, 0.7, 3.8, 1.2041, speed, t_ijk_R)

        # Update total energy consumption and total travel time
        total_energy += energy_this_period
        travel_time += t_ijk_R
        # print('travel_time', travel_time)

        # Update remaining distance
        remaining_distance -= speed * t_ijk_R

        if remaining_distance <= 0:
            break
        else:
            departure_time1 = period_end

    return total_energy, travel_time, speed


# total_distance_matrix从pd.DataFrame中导入数据
def insert_cs(dataframe, initial_Routes, total_distance_matrix, departure_time2):
    final_Routes = initial_Routes.copy()
    Q_max = 40
    Q = Q_max
    service_time = 10 / 60  # 10 minutes converted to hours
    # service_time = Fraction(10, 60)  # 10 minutes converted to hours
    departure_time3 = departure_time2
    total_charging_time = 0
    total_travel_time = 0
    # 计算总demand
    demands = [pd.to_numeric(dataframe.iloc[node]['demand'], errors='coerce') for node in initial_Routes]
    load = sum(demand for demand in demands if pd.notnull(demand))
    # 充电站节点
    CS_list = dataframe[dataframe['Type'] == 'f'].index.tolist()

    states = []  # Track the status of each node
    i = 0
    while i < len(final_Routes) - 1:
        dis = total_distance_matrix.at[final_Routes[i], final_Routes[i + 1]]
        energy, t_ijk, speed = energy_time(dis, departure_time3, load)
        total_travel_time += t_ijk
        # print(energy)
        departure_time3 += t_ijk + service_time
        Q -= energy

        states.append((i, Q, departure_time3, load, speed, final_Routes.copy()))

        if Q < 0:
            # When the power is not enough to reach the next node, start backtracking to find a suitable location to plug in the charging station
            inserted = False
            for prev_state in reversed(states[:-1]):
                prev_i, prev_Q, prev_time3, prev_load, prev_speed, prev_route = prev_state
                # Calculate the amount of electricity needed to get to the nearest charging station and select the nearest charging station
                cs_dis = [total_distance_matrix.at[final_Routes[prev_i], cs] for cs in CS_list]
                min_distance = min(cs_dis)
                min_distance_index = cs_dis.index(min_distance)
                cs_minIndex = CS_list[min_distance_index]

                cs_time = min_distance / prev_speed
                prev_energy = calculate_energy_consumption(1.184692, 1.112434, 9.8, 0,
                                                           0.012, 3000, prev_load, 0.7, 3.8, 1.2041, prev_speed,
                                                           cs_time)
                if prev_Q - prev_energy >= 0:
                    q_ik = prev_Q - prev_energy
                    charging_time = calculate_charging_time(q_ik)
                    total_charging_time += charging_time

                    # 找到最近的充电站并插入
                    # print('cs_min', cs_minIndex)
                    prev_route.insert(prev_i + 1, cs_minIndex)
                    final_Routes = prev_route
                    Q = Q_max  # 充电后电量恢复最大值
                    departure_time3 = states[prev_i + 1][2] + charging_time
                    load = states[prev_i + 1][3]  # update load
                    inserted = True
                    break

            if inserted:
                i = prev_i + 1  # Continue from the next node that plugs into the charging station
                continue  # Continue the main cycle
        i += 1
    return final_Routes, total_travel_time, total_charging_time


def total_cost(dataframe, final_routes, total_travel_time, total_charging_time, c1=120, c2=0.5, c3=0.3, c4=0.6):
    # print(final_routes)
    # dispatch = 0
    # for i in final_routes[:-1]:
    #     if dataframe.loc[i, 'Type'] == 'd' and dataframe.loc[i+1, 'Type'] == 'c':
    #         dispatch += 1
    dispatch_cost = c1 * 1

    travel_cost = c2 * 60 * total_travel_time

    customer_num = 0
    for i in final_routes:
        if dataframe.loc[i, 'Type'] == 'c':
            customer_num += 1
    service_cost = c3 * 10 * customer_num
    charging_cost = c4 * total_charging_time
    return dispatch_cost, travel_cost, service_cost, charging_cost


def update_pheromone(pheromone_matrix, customer_lists, AntRoutes, ant_costs, elite_ants, rho=0.9, f=10, epsilon=0.8,
                     t_min=0.05, t_max=2):
    # Create a copy of the pheromone matrix to apply decay
    decayed_pheromone_matrix = np.copy(pheromone_matrix) * (1 - rho)
    # Initialize a matrix to accumulate pheromone increments
    delta_pheromone_matrix = np.zeros_like(pheromone_matrix)

    # Create a mapping dictionary that maps the customer numbers in the customer_list to the index of the pheromone matrix
    customer_to_index = {customer: index for index, customer in enumerate(customer_lists)}

    # Accumulate pheromone increments for all ants
    for ant_id, path in AntRoutes.items():
        delta_pheromone = f / ant_costs[ant_id]
        for i in range(len(path) - 1):
            i_node, j_node = path[i], path[i + 1]
            delta_pheromone_matrix[customer_to_index[i_node]][customer_to_index[j_node]] += delta_pheromone

    # Add pheromone increment for the elite ant
    elite_ant_id, elite_ant_cost = elite_ants
    elite_delta_pheromone = epsilon * (f / elite_ant_cost)
    elite_path = AntRoutes[elite_ant_id]
    for i in range(len(elite_path) - 1):
        i_node, j_node = elite_path[i], elite_path[i + 1]
        delta_pheromone_matrix[customer_to_index[i_node]][customer_to_index[j_node]] += elite_delta_pheromone

    # Update the pheromone matrix by adding the increments to the decayed matrix
    new_pheromone_matrix = decayed_pheromone_matrix +  delta_pheromone_matrix

    # Ensure pheromone levels are within the specified bounds
    new_pheromone_matrix = np.clip(new_pheromone_matrix, t_min, t_max)

    return new_pheromone_matrix


def check_termination(Iter, maxIter):
    if Iter <= maxIter:
        return False  # Continue the algorithm
    else:
        return True  # Terminate and output the final optimized result






if __name__ == "__main__":
    df = pd.read_csv(
        r'C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\fig 3\c102_21.txt',
        sep=r'\s+')
    total_distance_matrix_df1 = pd.read_excel(
        r'C:\Users\12149\OneDrive - Universitatea Babeş-Bolyai\Desktop\EVRP_Datasets\Txt\evrptw_instances_LijunFan\fig 3\c102_21.xlsx')


    initial_Route = [0, 96, 25, 24, 92, 99, 104, 109, 117, 121, 120, 91, 102, 110, 119, 22, 26, 97, 114, 94, 107, 113, 23, 98, 116, 105, 103, 118, 108, 106, 111, 115, 112, 0]

    add_cs, travel_time, chargingtime = insert_cs(df, initial_Route, total_distance_matrix_df1, 0)
    print(add_cs)
    # for i in range(len(initial_Route)-1):
    #     print(total_distance_matrix_df1[i][i+1])


