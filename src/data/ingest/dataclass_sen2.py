import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import src.sdata.worldfloods.configs
import rasterio

# each object from class will wrap around one file
# each class will call the get function to grab an object
# object for 1 .SAFE file for 97 files each have 12 bands

# Which file type does modeling need? -> TIF for old time or .ZARR
# Use dataclass for WorldFloods bucket and then transfer
# to new bucket


@dataclass
class Sentinel2WF:
    folder_path: str
    filename: str
    file_format: str
    # for data versioning include attributes with respect to
    # the date the data was loaded (meta of metadata)
    # we want to trace the data that was used for a
    # model.
    load_date: datetime = field(default=datetime.now()) # metameta
    source_system: str = field(default="Not Specified") # metameta (raw, staging, datamart)
    source_resource_indicator # payload points to where data is?

    def __init__(self, folder_path: str, filename: str, file_format: str):
        pass

    def get_s2path_meta(self):
        return #folder_path+'/S2metadata/' + filename + '.geojson'

    def get_s2path_data(self):
        if (file_format == '.zarr'):
            pass
        else:
            return #folder_path+ '/S2/' + filename + '.tif'


#define in utils: convert_tif2zarr
    



    def get_s2_bandN(self, filepath_handle: str, bandN: str):
        return file_path_handle + bandN + "bandName





# S2A_MSIL1C_20181007T144101_N0206_R039_T22VDM_20181007T164836.SAFE/GRANULE/L1C_T22VDM_A017196_20181007T144302/IMG_DATA/T22VDM_20181007T144101_B01.jp2



