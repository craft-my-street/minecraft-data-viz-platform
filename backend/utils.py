
import os
import sqlite3
# from tabulate import tabulate
from datetime import datetime
import math
import json
import pandas as pd
import folium
from folium.plugins import HeatMap
from folium.plugins import TimestampedGeoJson
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re
import yaml
from profanity_check import predict_prob
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_style('whitegrid')


# variables
chat_columns = ["time", "x", "y", "z", "message", "username", "world_name"]
core_columns = ['time', 'x', 'y', 'z', 'data', 'meta', 'blockdata', 'rolled_back', 'user_name', 'material', 'world_name', 'action_name']
location_columns = ["x", "y", "z", "username", "time"]

# Get a list of named colors from matplotlib
colors = plt.cm.colors.cnames
# remove all light colors
colors = {k:v for k, v in colors.items() if 'light' not in k}
# remove white and black
colors.pop('white', None)
colors = {k:v for k, v in colors.items() if 'dark' in k}
# Print the list of colors
COLORS = list(colors.keys())


# carbon stock

def assess_carbon_stock(tree_type, volume):
    """
    Assess carbon stock for a given tree type.

    Args:
    - tree_type (str): Type of tree (spruce, oak, birch).
    - volume (float): Volume of the tree in cubic meters.

    Returns:
    - carbon_stock (float): Carbon stock in kilograms.
    """

    # Carbon content per unit volume for each tree type (in kg/m^3)
    carbon_content = {
        'spruce': 0.42,
        'oak': 0.52,
        'birch': 0.48
    }

    if tree_type.lower() not in carbon_content:
        raise ValueError("Invalid tree type. Available tree types: spruce, oak, birch")

    # Calculate carbon stock
    carbon_stock = volume * carbon_content[tree_type.lower()]

    return carbon_stock


# functions
def open_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def load_config(config_file):
    config = open_config(config_file)
    experiments = {}

    for experiment in config['Experiments']:
        experiments[experiment['name']] = experiment
    
    servers = {}
    for server in config['servers']:
        servers[server['name']] = server
    
    return experiments, servers


# def get_server(workshop_id):
#     if workshop_id not in config:
#         return None
#     return servers[config[workshop_id]['server']]

def tounixtime(date):
    return int(date.timestamp())


#   date : 08-04-2024
def tounixtime_2(date):
    return int(datetime.strptime(date, '%d-%m-%Y').timestamp())


def clean_chat(text):
    # Remove profanity
    def replace(match):
        return '*' * len(match.group())

    return re.sub(r'\b\w+\b', lambda x: replace(x) if predict_prob([x.group()])[0] > 0.5 else x.group(), text)


def execute_query(db_path, world_id, timestamp):
    try:
        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Define the SQL query
        sql_query = f"""
            SELECT DISTINCT
                co_block.time, co_block.x, co_block.y, co_block.z, co_block.data, co_block.meta, co_block.blockdata, co_block.rolled_back,
                co_user.user AS user_name, co_material_map.material AS material, co_world.world AS world_name,
                CASE co_block.action
                    WHEN 0 THEN 'break'
                    WHEN 1 THEN 'place'
                    WHEN 2 THEN 'interaction/click'
                    WHEN 3 THEN 'death/kill'
                    ELSE 'unknown'
                END AS action_name
            FROM co_block
            JOIN co_user ON co_block.user = co_user.id
            JOIN co_material_map ON co_block.type = co_material_map.id
            JOIN co_world ON co_block.wid = co_world.id
            WHERE co_user.user NOT LIKE '#%' AND
                co_block.action IN (0, 1, 2, 3) AND
                co_world.world = '{world_id}' AND
                co_block.time >= {timestamp}
            ORDER BY co_block.rowid DESC;
        """
# WHERE co_user.user NOT LIKE '#%' AND co_block.wid = 2
# WHERE co_block.ACTION = 0 
                # Execute the query
        cursor.execute(sql_query)

        # Fetch the results
        results = cursor.fetchall()
        print(f"[INFO] {len(results)} records were fetched!")

        # Fetch the column names
        column_names = [description[0] for description in cursor.description]

        # Display the results as a table
        # print(tabulate(results, headers=column_names, tablefmt="pretty"))
    

    except sqlite3.Error as e:
        print("SQLite error:", e)
    finally:
        # Close the database connection
        if connection:
            connection.close()
    return results, column_names


class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class GeoConverter:
    EARTH_RADIUS = 6371000.0  # Earth radius in meters

    def __init__(self, ingameCenter, xzScale, rotation, mapCenter):
        self.ingameCenter = ingameCenter
        self.xzScale = xzScale
        self.rotation = rotation
        self.mapCenter = mapCenter

    def deg(self, radians):
        return math.degrees(radians)

    def XZToLatLon(self, x, z):
        x -= self.ingameCenter.x
        z -= self.ingameCenter.y

        x /= self.xzScale
        z /= -self.xzScale

        x2 = x * math.cos(-self.rotation) - z * math.sin(-self.rotation)
        z2 = x * math.sin(-self.rotation) + z * math.cos(-self.rotation)

        x = x2
        z = z2

        rho = math.sqrt(x * x + z*z)
        c = math.asin(rho / self.EARTH_RADIUS)
        cosC = math.cos(c)
        sinC = math.sin(c)
        cosPhi0 = math.cos(self.mapCenter.x)
        sinPhi0 = math.sin(self.mapCenter.x)

        lat = self.mapCenter.x if rho == 0 else math.asin(cosC * sinPhi0 + (z * sinC * cosPhi0) / rho)
        lon = self.mapCenter.y + math.atan2(x * sinC, rho * cosC * cosPhi0 - z * sinC * sinPhi0)

        return self.deg(lat), self.deg(lon)#Vector2D(self.deg(lat), self.deg(lon))

def build_user_action_graph(df, result_dir):
    plt.figure(figsize=(20, 6))
    sns.countplot(data=df, x='user_name', hue='action_name')

    plt.savefig(f"{result_dir}/actions_by_user.png")
    # plt.show()


def filter_data(df):
    # Define filter types
    filter_types = [x for x in df['material'].unique() if x.endswith('_log')]
    filter_types.extend([x for x in df['material'].unique() if x.endswith('_leaves')])

    # Group the dataframe by 'material', 'user_name', and 'time'
    clusters = df.groupby(['material', 'user_name', 'time'])

    # Initialize lists to store indices to keep and remove
    indices_to_keep = []
    indices_to_remove = []

    # Iterate over each group
    for (name, _, time), group in clusters:
        if name in filter_types and len(group) > 1:
            # Add indices to remove
            indices_to_remove.extend(group.index)

    # Print summary
    print(f"{len(df)} rows are in the dataframe.")
    print(f"{len(indices_to_remove)} rows will be removed.")
    print(f"{len(df) - len(indices_to_remove)} rows will remain.")

    # Remove rows from the dataframe
    df_filtered = df.drop(indices_to_remove)

    return df_filtered


def extract_time_user_msg(log):
    data = []
    for msg in log:
        try:
            msg.strip()
            time = msg.split(" ")[0][1:-1]
            # the first occurance starts with # is the thread number
            thread_number = re.search(r'#(.*?)/', msg).group(1)
            

            user = re.search(r'<(.*?)>', msg).group(1)
            msg = msg.split("> ")[1].strip()
            data.append((time, thread_number, user, msg))
        except:
            pass
            # print(msg.strip())
    return data