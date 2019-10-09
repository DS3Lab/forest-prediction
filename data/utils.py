import os
import math
from pyproj import Proj, transform
from rasterio.merge import merge

def deg2num(lon_deg, lat_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return lon_deg, lat_deg

def geodesic2spherical(x1, y1, inverse=False):
    """
    EPSG:4326 to EPSG:3857:
        x1: longitude
        y1: latitude
    EPSG:3857 to EPSG:4326:
        x1: x coordinate
        y1: y coordinate
    """
    if inverse:
        inProj = Proj(init='epsg:3857')
        outProj = Proj(init='epsg:4326')
    else:
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:3857')
    x2,y2 = transform(inProj, outProj, x1,y1)
    return x2, y2

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
