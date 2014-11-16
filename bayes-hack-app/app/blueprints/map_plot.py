from flask import g
import numpy as np
import pandas as pd
import pylab as pl
from mpl_toolkits.basemap import Basemap

# import MySQLdb as mdb

# con = mdb.connect('bayesimpact.soumet.com', 'root', 'bayeshack', 'bayes') #host, user, password, #database

def lat_lon():
    # Build sql query
    connection = g.db_engine.connect()
    # cur = con.cursor()
    query = "SELECT school_latitude,school_longitude FROM donorschoose_projects LIMIT 100000"
    result = connection.execute(query)
    lats, lons = [],[]
    for lat,lon in result:
        lats.append(float(lat))
        lons.append(float(lon))
    return np.array(lats), np.array(lons)

def make_map():
    return Basemap(projection='merc', lat_0=40, lon_0=-110, resolution='l',
                   area_thresh=10000.0,
                   llcrnrlon=-179, urcrnrlon=-60, llcrnrlat=10, urcrnrlat=75)

def draw_base_map(map):
    pl.clf()
    # map.bluemarble()
    map.drawcoastlines()
    map.drawstates(color='green')
    map.drawrivers(color='blue')

def school_map(map=None):
    if map is None: map = make_map()
    lat1, lon1 = lat_lon()
    pl.clf()
    draw_base_map(map)
    return map.plot(lon1,lat1, 'ro', latlon=True)

