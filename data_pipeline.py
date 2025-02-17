import pandas as pd
import geopandas as gpd
import snowflake.connector
import logging
import os
import argparse

logging.basicConfig(format="%(levelname)s %(asctime)s %(message)s",
                    datefmt="%H:%M:%S", level = logging.INFO)

# Have the user specify the input files
# The shoreline file we're using is the coarsest one I found to speed up the feature engineering, can be changed out at the cost of increased time
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data',
    type = str,
    required = True,
    help = 'pathway to the parquet file to upload'
    )

parser.add_argument(
    '--shoreline',
    type = str,
    default='./ne_110m_coastline.zip',
    help = 'pathway to the shoreline file to use, defaults to the 110m coastline supplied in repo'
    )

args = parser.parse_args()

logging.info("Reading in the provided files")
try:
    df = pd.read_parquet(args.data)
    world = gpd.read_file(args.shoreline)
except Exception as e:
    logging.error("Error while opening file")
    logging.error(f"{e}")

# Use the feature engineering as described in the EDA notebook, prepare the dataframe for upload
logging.info("Preprocessing the data, this will take a moment")
df["timestamp_utc"] = pd.to_datetime(df["ts_pos_utc"])
df.rename(columns={'course':'heading'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df.dropna(inplace=True)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
gdf["distance_from_coast"] = gdf["geometry"].apply(lambda x: min(x.distance(coastline) for coastline in world["geometry"]))

def change_in_speed(x):
    return x.max() - x.min()

df_deltav = []
for mmsi, df_group in gdf[["mmsi","timestamp_utc","speed"]].groupby("mmsi"):
    df_resampled = df_group.resample("1h", on="timestamp_utc").apply(change_in_speed).dropna(axis=0)
    df_resampled['mmsi'] = mmsi
    df_resampled.reset_index(inplace=True)
    df_resampled.rename(columns={"speed":"change_in_speed","timestamp_utc":"rounded_hours"}, inplace=True)
    df_deltav.append(df_resampled)

df_deltav = pd.concat(df_deltav, ignore_index=True)
gdf["merging_hour"] = gdf["timestamp_utc"].dt.floor('H')
gdf = gdf.merge(df_deltav, left_on=["mmsi","merging_hour"], right_on=["mmsi","rounded_hours"])
df = gdf[['mmsi','vessel_class','timestamp_utc','speed', 'heading', 'latitude', 'longitude', 'distance_from_coast', 'change_in_speed','fishing']]

logging.info("Retrieving Snowflake credentials")
try:
    scn = snowflake.connector.connect(
        user = os.getenv("SNOWFLAKE_USER"),
        password = os.getenv("SNOWFLAKE_PASSWORD"),
        account = os.getenv("SNOWFLAKE_ACCOUNT")
    )
    scs = scn.cursor()
except Exception as e:
    logging.error("Error establishing Snowflake connection")
    logging.error(f"{e}")

logging.info("Creating Snowflake architecture")
scs.execute('CREATE WAREHOUSE IF NOT EXISTS geocore_warehouse')
scs.execute('CREATE DATABASE IF NOT EXISTS fishing_db')
scs.execute('USE DATABASE fishing_db')
scs.execute('CREATE SCHEMA IF NOT EXISTS fishing_schema')

scs.execute('USE WAREHOUSE geocore_warehouse')
scs.execute('USE DATABASE fishing_db')
scs.execute('USE SCHEMA fishing_schema')
sql = ("CREATE OR REPLACE TABLE fishing_raw_data"
       " (mmsi integer, vessel_class string, timestamp_utc timestamp_ntz, speed float, heading float, latitude float, longitude float, distance_from_coast float, change_in_speed float, fishing integer)")
scs.execute(sql)

logging.info("Inserting data")
# This is a very slow way of inserting data, apparently the built in `write_pandas` method is much faster, however something with my settings won't allow me to do it (RSA key not allowed)
# success, nchunkc, nrows, _ = write_pandas(scn, df, 'fishing_raw_data')
df["timestamp_utc"] = df["timestamp_utc"].dt.strftime('%Y-%m-%d %H:%M:%S')
table_rows = str(list(df.itertuples(index=False, name=None))).replace('[','').replace(']','')
scs.execute(f"INSERT INTO fishing_raw_data VALUES {table_rows}") 

sql = "SELECT COUNT(fishing) FROM fishing_raw_data"
scs.execute(sql)
row = scs.fetchone()
logging.info(f"Successfully uploaded {row[0]} rows of data")

scs.close()
scn.close()

