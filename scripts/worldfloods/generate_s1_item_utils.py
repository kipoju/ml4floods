# import all the necessary for the function below

import os
import warnings
import traceback
import sys
from typing import Tuple, Optional, Dict, Callable, Any

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ml4floods.data import utils
from ml4floods.data.ee_download import process_metadata
# from ml4floods.data.worldfloods.create_worldfloods_dataset import best_s2_match
from datetime import datetime, timezone
import rasterio
from ml4floods.data.worldfloods.configs import BANDS_S2
import tqdm
from ml4floods.data.create_gt import CLOUDS_THRESHOLD, BRIGHTNESS_THRESHOLD, get_brightness, compute_water, read_s2img_cloudmask
from skimage.morphology import binary_opening, disk
from ml4floods.data import save_cog

from georeader.rasterio_reader import RasterioReader
from ml4floods.data.worldfloods.create_worldfloods_dataset import _copy

def best_s2_match(metadatas2:pd.DataFrame, floodmap_date:datetime) -> Tuple[Any, datetime]:
    """
    Return s2 date posterior to the floodmap_date

    Args:
        metadatas2:
        floodmap_date:

    Returns:

    """
    index = None
    s2_date = None
    floodmap_date = floodmap_date.replace(tzinfo=timezone.utc)
    for tup in metadatas2[metadatas2.s2available].itertuples():
        date_img = tup.datetime
        if (floodmap_date < date_img) or ((floodmap_date - date_img).total_seconds() / 3600. < 10):
            if s2_date is None:
                s2_date = date_img
                index = tup.Index
            else:
                if s2_date > date_img:
                    s2_date = date_img
                    index = tup.Index
    return index, s2_date


def read_s1img(s1_image_path: str, window: Optional[rasterio.windows.Window] = None) -> np.ndarray:
    with utils.rasterio_open_read(s1_image_path) as s1_rst:
        s1_img = s1_rst.read([1,2], window=window)
    return s1_img

def best_s1_match(path_aoi: str, floodmap_date: datetime) -> str:
    
    fs = utils.get_filesystem(path_aoi)
    path_csv = os.path.join(path_aoi, "S1", "s1info.csv").replace("\\", "/")
    if path_csv.startswith("gs"):
        with fs.open(path_csv, "r") as fh:
            datas1 = pd.read_csv(fh)
    else:
        datas1 = pd.read_csv(path_csv)
        
    datas1["datetime"] = datas1.datetime.apply(lambda x: datetime.fromisoformat(x).replace(tzinfo=timezone.utc))

    datas1["names1file"] = datas1.datetime.apply(lambda x: x.strftime("%Y-%m-%d"))
    datas1["s1available"] = datas1.names1file.apply(lambda x: fs.exists(os.path.join(os.path.dirname(path_csv),
                                                                                     x +".tif")))
    index = None
    s1_date = None
    floodmap_date = floodmap_date.replace(tzinfo=timezone.utc)
    for tup in datas1[datas1.s1available].itertuples():
        date_img = tup.datetime
        if (floodmap_date < date_img) or ((floodmap_date - date_img).total_seconds() / 3600. < 10):
            if s1_date is None:
                s1_date = date_img
                index = tup.Index
            else:
                if s1_date > date_img:
                    s1_date = date_img
                    index = tup.Index
                    
    return  index, s1_date


def worldfloods_v3_gcp_paths(main_path: str) -> Tuple[gpd.GeoDataFrame, Optional[str], Optional[str], Dict, str]:
    """
    Given a pickle file in "gs://ml4cc_data_lake/{prod_dev}/1_Staging/WorldFloods it returns the rest of the files to
    create the full worldfloods registry (the corresponding floodmap, cloud probability and permanent water)

    Args:
        main_path: path to .piclke file in bucket

    Returns:
        Locations of corresponding  floodmap, cloud probability,  permanent water, metadata_floodmap and S2 image

    """
    fs = fsspec.filesystem("gs", requester_pays=True)
    assert fs.exists(main_path), f"File {main_path} does not exists"

    meta_floodmap = utils.read_pickle_from_gcp(main_path)

    # path to floodmap path
    floodmap_path = main_path.replace("/flood_meta/", "/floodmap_edited/").replace(".pickle", ".geojson")
    if not fs.exists(floodmap_path):
        floodmap_path = floodmap_path.replace("/floodmap_edited/", "/floodmap/")

    assert fs.exists(floodmap_path), f"Floodmap not found in {floodmap_path}"

    path_aoi = os.path.dirname(os.path.dirname(floodmap_path))

    # open floodmap with geopandas
    floodmap = utils.read_geojson_from_gcp(floodmap_path)
    floodmap_date = meta_floodmap['satellite date']

    # create permenant water path
    permanent_water_path = os.path.join(path_aoi, "PERMANENTWATERJRC", f"{floodmap_date.year}.tif").replace("\\", "/")

    if not fs.exists(permanent_water_path):
        warnings.warn(f"Permanent water {permanent_water_path}. Will not be used")
        permanent_water_path = None

    csv_path = os.path.join(path_aoi, "S2", "s2info.csv")
    metadatas2 = process_metadata(csv_path, fs=fs)
    metadatas2 = metadatas2.set_index("names2file")

    assert any(metadatas2.s2available), f"Not available S2 files for {main_path}. {metadatas2}"

    # find corresponding s2_image_path for this image
    index, s2_date = best_s2_match(metadatas2, floodmap_date)

    assert s2_date is not None, f"Not found valid S2 files for {main_path}. {metadatas2}"

    s2_date = s2_date.replace(tzinfo=None)
    assert (s2_date - floodmap_date).total_seconds() / (3600. * 24) < 10, \
        f"Difference between S2 date {s2_date} and floodmap date {floodmap_date} is larger than 10 days"

    # add exact metadata from the csv file
    meta_floodmap["s2_date"] = s2_date
    meta_floodmap["names2file"] = index
    meta_floodmap["cloud_probability"] = metadatas2.loc[index, "cloud_probability"]
    meta_floodmap["valids"] = metadatas2.loc[index, "valids"]

    s2_image_path = os.path.join(path_aoi, "S2", index+".tif").replace("\\", "/")
    
    # add s1 image path:
    index, date_s1 = best_s1_match(path_aoi, floodmap_date)
    s1_image_path = os.path.join(path_aoi, "S1", datetime.strftime(date_s1, '%Y-%m-%d')+".tif").replace("\\", "/")
    assert fs.exists(s1_image_path), f"Not S1 images found"

    # Add cloud_probability if exists in edited
    cm_edited = s2_image_path.replace("/S2/", "/cmedited_vec/").replace(".tif", ".geojson")
    if not fs.exists(cm_edited):
        cm_edited = s2_image_path.replace("/S2/", "/cmedited/")
        if not fs.exists(cm_edited):
            cm_edited = None

    return floodmap, cm_edited, permanent_water_path, meta_floodmap, s1_image_path, s2_image_path

def _generate_gtv3_fromarray(
    s2_img: np.ndarray,
    s1_img: np.ndarray,
    cloudprob: np.ndarray,
    water_mask: np.ndarray,
    custom_clouds: bool=False,
) -> np.ndarray:
    """

    Generate Ground Truth of WorldFloods Extra (multi-output binary classification problem)

    Args:
        s2_img: (C, H, W) array
        s1_img: (2, H, W) array
        cloudprob: (H, W) array
        water_mask: (H, W) array {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
        custom_clouds: whether it uses a custom gt or not.
            If it is not custom it will mask the bright cloud pixels in the s2image in the water mask.

    Returns:
        (2, H, W) np.uint8 array where:
            First channel encodes {0: invalid, 1: clear, 2: cloud}
            Second channel encodes {0: invalid, 1: land, 2: water}

        A pixel is set to invalid if it's invalid in the water_mask layer or invalid in the s2_img (all values to zero)

    """
    # Mark as invalid if it is out of swath for both satellites (for Sentinel-2 it includes 
    # thick clouds), and if it is outside of the area of interest bounds, i.e. is invalid in the water mask


    invalids_s2 = np.any(np.isnan(s2_img), axis=0) | np.all(s2_img[:len(BANDS_S2)] == 0, axis=0) | (water_mask == -1)
    clouds = cloudprob > CLOUDS_THRESHOLD
    if custom_clouds:
        # set to invalid in watergt clouds in gt (assume manually added to include only thick clouds!)
        open_clouds = clouds
    else:
        # set to invalid in watergt for bright clouds
        brightness = get_brightness(s2_img) # (H, W)
        clouds &= (brightness >= BRIGHTNESS_THRESHOLD)

        # binary opening of bright clouds
        open_clouds = binary_opening(clouds, disk(3)).astype(np.bool_)

    invalids_s2 |= open_clouds
    invalids_s1 = np.any(np.isnan(s1_img), axis=0) | (water_mask == -1)
    invalids = invalids_s2 | invalids_s1 # if OR Sentinel-2 cloud masks will be masked, since its the source of the gt, makes sense.

    # Set watermask values for compute stats
    watergt = np.ones(water_mask.shape, dtype=np.uint8)  # whole image is 1
    watergt[water_mask >= 1] = 2  # only water is 2
    watergt[invalids] = 0
    
    invalids_return =  np.zeros(invalids.shape, dtype=np.uint8)
    invalids_return[invalids_s2] = 1
    invalids_return[invalids_s1] = 2
    stacked_gt = np.stack([watergt, invalids_return], axis=0)

    return stacked_gt

## GENERATE GT FUNCTIONS

def generate_water_gt(
    s2_local_image_path: str,
    gt_binay_path: str,
    s1_image_path: str,
    floodmap: gpd.GeoDataFrame,
    window: Optional[rasterio.windows.Window] = None,
    keep_streams: bool = False,
    permanent_water_image_path: Optional[str] = None,
    cloudprob_image_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    New ground truth generating function for multioutput binary classification

    Args:
        s2_image_path:
        floodmap:
        metadata_floodmap: Metadata of the floodmap (if satellite is optical will mask the land/water GT)
        window:
        keep_streams:
        permanent_water_image_path:
        cloudprob_image_path:

    Returns:
        gt (np.ndarray): (2, H, W) np.uint8 array where:
            First channel encodes the cloud GT {0: invalid, 1: clear, 2: cloud}
            Second channel encodes the land/water GT {0: invalid, 1: land, 2: water}
        meta: dictionary with metadata information

    """

    # =========================================
    # Generate Cloud Mask given S2 Data
    # =========================================
    s2_img = RasterioReader(s2_local_image_path, indexes = list(range(len(BANDS_S2))), window_focus=window).values
    
    gt_binary = RasterioReader(gt_binay_path, window_focus=window).values[0]
    
    cloud_mask = (gt_binary == 2).astype(np.uint8)

    custom_cloud_mask = True # it will directly use the cloud mask from the gt
    
    s1_img = read_s1img(s1_image_path, window=window)

    water_mask = compute_water(
        s2_local_image_path,
        floodmap,
        window=window,
        permanent_water_path=permanent_water_image_path,
        keep_streams=keep_streams,
    )

    # # TODO this should be invalid if it is Sentinel-2 and it is exactly the same date ('satellite date' is the same as the date of retrieval of s2tiff)
    # if metadata_floodmap["satellite"] == "Sentinel-2":
    #     invalid_clouds_threshold = 0.5
    # else:
    #     invalid_clouds_threshold = None

    gt = _generate_gtv3_fromarray(
        s2_img,
        s1_img,
        cloudprob=cloud_mask,
        water_mask=water_mask,
        custom_clouds=custom_cloud_mask,
    )

    return gt


def generate_s1_item(main_path:str, output_path:str, file_name:str,
                  overwrite:bool=False, pbar:Optional[tqdm.tqdm]=None,
                  gt_fun:Callable=None, delete_if_error:bool=True,
                  paths_function:Callable=worldfloods_v3_gcp_paths) -> bool:
    """

    Generates an "element" of the WorldFloods dataset with the expected naming convention. An "element" is a set of files
    that will be used for training the WorldFloods model. These files are copied in the `output_path` folowing the naming
    convention coded in `worldfloods_output_files` function.
     These files are:
    - shp vector floodmap. (subfolder floodmap)
    - tiff permanent water (subfolder PERMANENTWATERJRC)
    - tiff gt
    - tiff S2
    - tiff cloudprob
    - json with metadata info of the ground truth

    Args:
        main_path: Path to main object. The process will search for other relevant information to create all the
        aforementioned products.
        output_path: Folder where the item will be written. See fun worldfloods_output_files for corresponding output file naming convention.
        overwrite: if False it will not overwrite existing products in path_write folder.
        file_name: Name of the file to be saved (e.g. in S2/gt/floodmap/ data will be saved with this name and the corresponding extension .tif, .geojson)
        pbar: optional progress bar with method description.
        gt_fun: one of ml4floods.data.create_gt.generate_land_water_cloud_gt or ml4floods.data.create_gt.generate_water_cloud_binary_gt.
        This function determines how the ground truth is created from the input products.
        delete_if_error: whether to delete the generated files if an error is risen
        paths_function: function to get the paths of the files for the given main_path

    Returns:
        True if success in creating all the products

    """

    fs = fsspec.filesystem("gs", requester_pays=True)

    try:
        # Check if output products exist before reading from the bucket
        gt_path_dest = os.path.join(output_path,"gtwater",file_name+".tif").replace("\\", "/")
        if not overwrite and os.path.exists(gt_path_dest):
            return True
        os.makedirs(os.path.join(output_path,"gtwater"),exist_ok=True)
        
        # Get input files and check that they all exist
        floodmap, cloudprob_path, permanent_water_path, metadata_floodmap, s1_image_path, s2_image_path = paths_function(main_path)
        ## TODO incorporate S1 to scratch generation
        s2_image_path_local = os.path.join(output_path,"S2",file_name+".tif").replace("\\", "/")
        gt_local_path = os.path.join(output_path,"gt",file_name+".tif").replace("\\", "/")
        
    except Exception:
        warnings.warn(f"File {main_path} problem when computing input/output names")
        traceback.print_exc(file=sys.stdout)
        return False
    try:
        # generate gt, gt meta and copy all files to path_write
        fsdest = utils.get_filesystem(gt_path_dest)

        if not fsdest.exists(gt_path_dest) or overwrite:
            if pbar is not None:
                pbar.write(f"Generating Ground Truth {file_name}...")


            gt = generate_water_gt(
                s2_image_path_local,
                gt_local_path,
                s1_image_path,
                floodmap,
                keep_streams=True,
                cloudprob_image_path=cloudprob_path, # Could be None!
                permanent_water_image_path=permanent_water_path,  # Could be None!
            )

            if pbar is not None:
                pbar.write(f"Saving GT {file_name}...")

            with rasterio.open(s2_image_path_local) as src:
                transform = src.transform
                crs = src.crs
                
            save_cog.save_cog(gt, gt_path_dest,
                                {"crs": crs, "transform": transform ,"RESAMPLING": "NEAREST",
                                "compress": "lzw", "nodata": 0}, # In both gts 0 is nodata
                                descriptions=["invalid/land/water/", "s2swath/s1swath"])
            
        # Copy S1 image
        s1_image_path_dest = os.path.join(output_path,"S1",file_name+".tif").replace("\\", "/")
        if not fsdest.exists(s1_image_path_dest) or overwrite:
            if pbar is not None:
                pbar.write(f"Saving S1 image {file_name}...")
            
            _copy(s1_image_path, s1_image_path_dest, fs)
            
        # read metadata and add the date of Sentinel-1
        _, s1_date = best_s1_match(os.path.dirname(os.path.dirname(s1_image_path)), metadata_floodmap["satellite date"])
        metadata_floodmap["s1_date"] = s1_date
        from ml4floods.data.utils import write_json_to_gcp
        write_json_to_gcp(os.path.join(output_path,"meta",file_name+".json").replace("\\", "/"), metadata_floodmap)


    except Exception:
        warnings.warn(f"File input: {main_path} output S2 file: {gt_path_dest} problem when computing Ground truth")
        traceback.print_exc(file=sys.stdout)

        if not delete_if_error:
            return False


        return False

    return True