import os
import math
from datetime import datetime
import json, yaml, random, re
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString

from utils import Vector2D, GeoConverter
from utils import load_config, tounixtime, execute_query, filter_data, extract_time_user_msg, clean_chat
from utils import chat_columns, core_columns, location_columns, COLORS
from database import build_chatdata_query, build_corepro_query, build_location_query, fetch_data_from_mysql

status_codes = {
    "success": 200,
    "bad_request": 400,
    "not_found": 404,
    "internal_error": 500
}

VERSION = "v1"

app = FastAPI(
    title="DataViz Portal API",
    description="This API provides an interface for data visualization for Minecraft gameplay data.",
    version=VERSION,
    redocs_url="/docs",
    docs_url=None,
    contact={'name': 'Lasith Niroshan', 'url': 'https://lasith-niro.github.io/', 'email': 'lasith.kottawahewamanage@ucd.ie'},
    license_info={'name': 'not yet decided'},
)
app_id = f"/api/{VERSION}"

data_dir = "data"
conf_file = "config.yaml"
config, servers = load_config(conf_file)

gameplays = list(config.keys())

@app.get(f'{app_id}/test')
async def test():
    return {"status": "success", "message": "Data sharing API is up and running"}


@app.get(f'{app_id}/gameplays')
async def get_gameplays():
    return {"gameplays": gameplays, "len": len(gameplays), "status": status_codes["success"]}



def get_server(workshop_id):
    if workshop_id not in config:
        return None
    return servers[config[workshop_id]['server']]

def common_preprocess_steps(df: pd.DataFrame, configs: dict, base_dir: str, gameplay_id: str) -> pd.DataFrame:
    ingameCenter = Vector2D(0, 0)
    map_center_lat, map_center_lon = configs[gameplay_id]["map_center"]
    xzScale = configs[gameplay_id]["xzScale"]
    rotation = configs[gameplay_id]["rotation"]
    mapCenter = Vector2D(math.radians(map_center_lat), math.radians(map_center_lon))
    geo_converter = GeoConverter(ingameCenter, xzScale, rotation, mapCenter)
    df['lat'], df['lon'] = zip(*df.apply(lambda row: geo_converter.XZToLatLon(row['x'], row['z']), axis=1))
    df['time'] = df['time'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    user_map_file = os.path.join(base_dir, "user_map.json")
    with open(user_map_file, 'r') as f:
        user_map = json.load(f)
    return df, user_map

def preprocess_coredata(df: pd.DataFrame, configs: dict, base_dir: str, gameplay_id: str) -> pd.DataFrame:
    df, user_map = common_preprocess_steps(df, configs, base_dir, gameplay_id)
    df['user_name'] = df['user_name'].apply(lambda x: user_map[x])
    df['material'] = df['material'].apply(lambda x: x.split(":")[1])
    column_order = ['time', 'x', 'y', 'z', 'lat', 'lon', 'world_name', 'user_name', 'material', 'action_name', 'data', 'meta', 'blockdata', 'rolled_back']
    df = df.reindex(columns=column_order)
    return df

def preprocess_chatdata(df: pd.DataFrame, configs: dict, base_dir: str, gameplay_id: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df['cleaned_message'] = df['message'].apply(clean_chat).str.lower()
    df, user_map = common_preprocess_steps(df, configs, base_dir, gameplay_id)
    df['username'] = df['username'].replace(user_map)
    return df

def preprocess_locationdata(df: pd.DataFrame, configs: dict, base_dir: str, gameplay_id: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df, user_map = common_preprocess_steps(df, configs, base_dir, gameplay_id)
    df['username'] = df['username'].replace(user_map)
    return df

@app.get(f'{app_id}/gameplay')
async def get_gameplay(gameplay_id: str):
    gameplay_id = gameplay_id.strip()
    print(f"[INFO] Fetching data for gameplay id: {gameplay_id}")
    if gameplay_id not in gameplays:
        return {"status": "bad_request", "message": f"Gameplay id {gameplay_id} not found", "status_code": status_codes["bad_request"]}
    
    base_dir = os.path.join(data_dir, gameplay_id)
    if os.path.exists(base_dir):
        # send the data from csv files
        # pass
        print(f"[INFO] Data found on the local storage : {base_dir}/core.csv")
    else:
        os.makedirs(base_dir, exist_ok=True)
        server = get_server(gameplay_id)
        host, username, password, database = server['host'], server['username'], server['password'], server['database_name']
        print(f"[INFO] Fetching data from MySQL database: {host}, {username}, {database}")
        core_query = build_corepro_query(config, gameplay_id)
        chat_query = build_chatdata_query(config, gameplay_id)
        location_query = build_location_query(config, gameplay_id)

        core_data = fetch_data_from_mysql(host, username, password, database, core_query)
        chat_data = fetch_data_from_mysql(host, username, password, database, chat_query)
        location_data = fetch_data_from_mysql(host, username, password, database, location_query)

        core_df = pd.DataFrame(core_data, columns=core_columns)
        chat_df = pd.DataFrame(chat_data, columns=chat_columns)
        location_df = pd.DataFrame(location_data, columns=location_columns)

        # save user_map
        user_map_file = os.path.join(base_dir, "user_map.json")
        user_names = core_df['user_name'].unique()
        username_map = {name: f"Player_{i+1}" for i, name in enumerate(user_names)}
        with open(user_map_file, 'w') as f:
            json.dump(username_map, f)

        core_df = preprocess_coredata(core_df, config, base_dir, gameplay_id)
        chat_df = preprocess_chatdata(chat_df, config, base_dir, gameplay_id)
        location_df = preprocess_locationdata(location_df, config, base_dir, gameplay_id)

        core_df.to_csv(f"{base_dir}/core.csv", index=False)
        chat_df.to_csv(f"{base_dir}/chat.csv", index=False)
        location_df.to_csv(f"{base_dir}/location.csv", index=False)
        
        stats = {
            "core": len(core_df),
            "chat": len(chat_df),
            "location": len(location_df),
            "users" : len(user_names)
            # "paths": {
            #     "core": f"{base_dir}/core.csv",
            #     "chat": f"{base_dir}/chat.csv",
            #     "location": f"{base_dir}/location.csv"
            # }
        }

        output = {
            "game_analytics": True if len(core_df) > 0 else False,
            "chat_analytics": True if len(chat_df) > 0 else False,
            "location_analytics": True if len(location_df) > 0 else False
        }
        return {"status": "success", "message": output, "stats": stats, "status_code": status_codes["success"], "data" : {"host": host, "user": username, "password": password, "database": database}}

# send raw dataframe
@app.get(f'{app_id}/gameplay/df')
async def get_gameplay_df(gameplay_id: str, type:str)-> JSONResponse:
    gameplay_id = gameplay_id.strip()
    type = type.strip()

    if gameplay_id not in gameplays:
        return {"status": "bad_request", "message": f"Gameplay id {gameplay_id} not found", "status_code": status_codes["bad_request"]}
    
    if type not in ["core", "chat", "location"]:
        return {"status": "bad_request", "message": f"Invalid type {type}", "status_code": status_codes["bad_request"]}

    base_dir = os.path.join(data_dir, gameplay_id)
    df = pd.read_csv(f"{base_dir}/{type}.csv")

    return JSONResponse(content=df.to_json(), status_code=status_codes["success"])

# get filter input
@app.get(f'{app_id}/gameplay/filter')
async def get_filter_input(gameplay_id: str, filter: str):
    gameplay_id = gameplay_id.strip()
    filter = filter.strip()
    print(filter)
    return {"status": "success", "message": "Filter input received", "status_code": status_codes["success"]}


# user-interactions: actions_by_user
@app.get(f'{app_id}/gameplay/actions_by_user')
async def get_actions_by_user(gameplay_id: str, filter: str):
    gameplay_id = gameplay_id.strip()
    filter = filter.strip()
    filter = json.loads(filter)
    
    base_dir = os.path.join(data_dir, gameplay_id)
    if not os.path.exists(base_dir):
        return {"status": "bad_request", "message": f"Data for gameplay id {gameplay_id} not found\nFirst call /gameplay?gameplay_id=<name> to create a dataset.", "status_code": status_codes["bad_request"]}
    core_df = pd.read_csv(f"{base_dir}/core.csv")
    # count action_name by user_name
    if filter == "all":
        action_count = core_df["action_name"].value_counts()
        return {"status": "success", "action_count": action_count.to_dict(), "status_code": status_codes["success"]}
    else:
        filtered_filters = {key: value for key, value in filter.items() if value != ["all"]}
        filtered_df = core_df.copy()
        for key, values in filtered_filters.items():
            filtered_df = filtered_df[filtered_df[key].str.contains('|'.join(values), regex=True)]
        
        action_count = filtered_df["action_name"].value_counts()
        return {"status": "success", "action_count": action_count.to_dict(), "status_code": status_codes["success"]}
    
    
@app.get(f'{app_id}/gameplay/players')
async def get_players(gameplay_id: str):
    gameplay_id = gameplay_id.strip()
    base_dir = os.path.join(data_dir, gameplay_id)
    if not os.path.exists(base_dir):
        return {"status": "bad_request", "message": f"Data for gameplay id {gameplay_id} not found\nFirst call /gameplay?gameplay_id=<name> to create a dataset.", "status_code": status_codes["bad_request"]}
    core_df = pd.read_csv(f"{base_dir}/core.csv")
    return {"status": "success", "players": core_df['user_name'].unique().tolist(), "status_code": status_codes["success"]}

@app.get(f'{app_id}/gameplay/materials')
async def get_materials(gameplay_id: str):
    gameplay_id = gameplay_id.strip()
    base_dir = os.path.join(data_dir, gameplay_id)
    if not os.path.exists(base_dir):
        return {"status": "bad_request", "message": f"Data for gameplay id {gameplay_id} not found\nFirst call /gameplay?gameplay_id=<name> to create a dataset.", "status_code": status_codes["bad_request"]}
    core_df = pd.read_csv(f"{base_dir}/core.csv")
    return {"status": "success", "materials": core_df['material'].unique().tolist(), "status_code": status_codes["success"]}

@app.get(f'{app_id}/gameplay/actions')
async def get_actions(gameplay_id: str):
    gameplay_id = gameplay_id.strip()
    base_dir = os.path.join(data_dir, gameplay_id)
    if not os.path.exists(base_dir):
        return {"status": "bad_request", "message": f"Data for gameplay id {gameplay_id} not found\nFirst call /gameplay?gameplay_id=<name> to create a dataset.", "status_code": status_codes["bad_request"]}
    core_df = pd.read_csv(f"{base_dir}/core.csv")
    return {"status": "success", "actions": core_df['action_name'].unique().tolist(), "status_code": status_codes["success"]}

# actions-over-time
@app.get(f'{app_id}/gameplay/actions_over_time')
async def get_actions_over_time(gameplay_id: str, filter: str):
    gameplay_id = gameplay_id.strip()
    filter = filter.strip()
    filter = json.loads(filter)
    
    base_dir = os.path.join(data_dir, gameplay_id)
    if not os.path.exists(base_dir):
        return {"status": "bad_request", "message": f"Data for gameplay id {gameplay_id} not found\nFirst call /gameplay?gameplay_id=<name> to create a dataset.", "status_code": status_codes["bad_request"]}
    core_df = pd.read_csv(f"{base_dir}/core.csv")
    if filter == "all":
        action_count = core_df.groupby('time')['action_name'].value_counts().unstack().fillna(0)
        return {"status": "success", "action_count": action_count.to_dict(), "status_code": status_codes["success"]}
    else:
        filtered_filters = {key: value for key, value in filter.items() if value != ["all"]}
        filtered_df = core_df.copy()
        for key, values in filtered_filters.items():
            filtered_df = filtered_df[filtered_df[key].str.contains('|'.join(values), regex=True)]
        
        action_count = filtered_df.groupby('time')['action_name'].value_counts().unstack().fillna(0)
        return {"status": "success", "action_count": action_count.to_dict(), "status_code": status_codes["success"]}



# purge/gameplay_id
@app.get(f'{app_id}/purge')
async def purge_data(gameplay_id: str):
    gameplay_id = gameplay_id.strip()
    print(f"[INFO] Purging data for gameplay id: {gameplay_id}")
    if gameplay_id not in gameplays:
        return {"status": "bad_request", "message": f"Gameplay id {gameplay_id} not found", "status_code": status_codes["bad_request"]}
    
    base_dir = os.path.join(data_dir, gameplay_id)
    if not os.path.exists(base_dir):
        return {"status": "bad_request", "message": f"Data for gameplay id {gameplay_id} not found", "status_code": status_codes["bad_request"]}
    
    try:
        os.system(f"rm -rf {base_dir}")
        return {"status": "success", "message": f"Data for gameplay id {gameplay_id} purged successfully", "status_code": status_codes["success"]}
    except Exception as e:
        return {"status": "internal_error", "message": f"Error in purging data for gameplay id {gameplay_id}", "status_code": status_codes["internal_error"]}

# gameplay_id/all

# gameplay/user/<username>

if __name__ == '__main__':
    import uvicorn
    # setup autoreload on change
    # uvicorn.run('app.api:main', host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
