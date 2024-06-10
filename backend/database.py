import mysql.connector
from utils import tounixtime

# queries
def build_chatdata_query(config, gameplay_id):
    chat_query = f"""
    SELECT DISTINCT
        cpro_chat.time, cpro_chat.x, cpro_chat.y, cpro_chat.z, cpro_chat.message,
        cpro_user.user AS username, cpro_world.world AS world_name
    FROM cpro_chat
    JOIN cpro_user ON cpro_chat.user = cpro_user.rowid
    JOIN cpro_world ON cpro_chat.wid = cpro_world.id
    WHERE 
        cpro_user.user NOT LIKE '#%' AND
        cpro_world.world = '{config[gameplay_id]["world_id"]}' AND
        cpro_chat.time >= {config[gameplay_id]["start_date"].timestamp()} AND
        cpro_chat.time <= {config[gameplay_id]["end_date"].timestamp()}
    ORDER BY cpro_chat.rowid DESC;
    """
    return chat_query

def build_corepro_query(config, gameplay_id):
    corepro_query = f"""
        SELECT DISTINCT
            cpro_block.time, cpro_block.x, cpro_block.y, cpro_block.z, cpro_block.data, cpro_block.meta, cpro_block.blockdata, cpro_block.rolled_back,
            cpro_user.user AS user_name, cpro_material_map.material AS material, cpro_world.world AS world_name,
            CASE cpro_block.action
                WHEN 0 THEN 'break'
                WHEN 1 THEN 'place'
                WHEN 2 THEN 'interaction/click'
                WHEN 3 THEN 'death/kill'
                ELSE 'unknown'
            END AS action_name
        FROM cpro_block
        JOIN cpro_user ON cpro_block.user = cpro_user.rowid
        JOIN cpro_material_map ON cpro_block.type = cpro_material_map.id
        JOIN cpro_world ON cpro_block.wid = cpro_world.id
        WHERE 
            cpro_user.user NOT LIKE '#%' AND
            cpro_block.action IN (0, 1, 2, 3) AND
            cpro_world.world = '{config[gameplay_id]["world_id"]}' AND
            cpro_block.time >= {config[gameplay_id]["start_date"].timestamp()} AND
            cpro_block.time <= {config[gameplay_id]["end_date"].timestamp()}
        ORDER BY cpro_block.rowid DESC;
                """
        
    return corepro_query

def build_location_query(config, gameplay_id):
    location_query = f"""SELECT x, y, z, username, time
                    FROM whimc_player_positions
                    WHERE
                    world = '{config[gameplay_id]["world_id"]}' AND
                    time >= '{config[gameplay_id]["start_date"].timestamp()}' AND
                    time <= '{config[gameplay_id]["end_date"].timestamp()}'
                """
    print(location_query)
    return location_query


# classes


# functions
def fetch_data_from_mysql(url, username, password, database, query):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host=url,
            user=username,
            password=password,
            database = database
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Execute the query
            cursor = connection.cursor()
            cursor.execute(query)

            # Fetch all rows
            rows = cursor.fetchall()

            return rows

    except mysql.connector.Error as error:
        print("Error while connecting to MySQL database:", error)

    finally:
        # Close connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL database connection closed")