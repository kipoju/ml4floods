
import sys
import warnings
import traceback

import tqdm
import pandas as pd
from datetime import datetime
from ml4floods.data import utils
from ml4floods.data.ee_download import process_metadata
import os

from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any
import geopandas as gpd
import fsspec
from ml4floods.data import save_cog

from ml4floods.data.create_gt import compute_water
import rasterio
import rasterio.windows
from rasterio import features
import numpy as np
from ml4floods.data.worldfloods.create_worldfloods_dataset import _copy

CLOUDPROB_PARENT_PATH = "worldfloods/tiffimages"
PERMANENT_WATER_PARENT_PATH = "worldfloods/tiffimages/PERMANENTWATERJRC"
META_FLOODMAP_PARENT_PATH = "worldfloods/tiffimages/meta"
WORLDFLOODS_V0_BUCKET = "ml4floods"


def worldfloods_v3_paths(main_path: str) -> Tuple[gpd.GeoDataFrame, Optional[str], Optional[str], Dict, str]:
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

    csv_path = os.path.join(path_aoi, "S1_GRD", "s1info.csv")
    metadatas2 = process_metadata(csv_path, fs=fs)
    metadatas2 = metadatas2.set_index("names2file")

    assert any(metadatas2.s2available), f"Not available S1 files for {main_path}. {metadatas2}"

    # find corresponding s2_image_path for this image
    index_pre, index_post, s1_date_pre, s1_date_post = pre_post_flood_image(metadatas2, floodmap_date)

    assert s1_date_pre is not None and  s1_date_post is not None , f"Not found valid S1 files for {main_path}. {metadatas2}"

    assert (s1_date_post - floodmap_date).total_seconds() / (3600. * 24) < 10, \
        f"Difference between S1 post date {s1_date_post} and floodmap date {floodmap_date} is larger than 10 days"
    # assert (floodmap_date - s1_date_pre ).total_seconds() / (3600. * 24) > 10, \
    #     f"Difference between S1 pre date {s1_date_pre} and floodmap date {floodmap_date} is larger than 10 days"

    # add exact metadata from the csv file
    meta_floodmap["s1_date_pre"] = s1_date_pre
    meta_floodmap["s1_date_post"] = s1_date_post
    meta_floodmap["names1pre"] = index_pre
    meta_floodmap["names1pre"] = index_post
    meta_floodmap["valids_pre"] = metadatas2.loc[index_pre, "overlap"]
    meta_floodmap["valids_post"] = metadatas2.loc[index_post, "overlap"]

    s1_image_pre_path = os.path.join(path_aoi, "S1_GRD", index_pre+".tif").replace("\\", "/")
    s1_image_post_path = os.path.join(path_aoi, "S1_GRD", index_post+".tif").replace("\\", "/")

    # # Add cloud_probability if exists in edited
    # cm_edited = s2_image_path.replace("/S2/", "/cmedited_vec/").replace(".tif", ".geojson")
    # if not fs.exists(cm_edited):
    #     cm_edited = s2_image_path.replace("/S2/", "/cmedited/")
    #     if not fs.exists(cm_edited):
    #         cm_edited = None

    # TODO Add way to handle same for S2
    
    return floodmap, permanent_water_path, meta_floodmap, s1_image_pre_path, s1_image_post_path

def worldfloods_v3_output_files(output_path:str, file_name:str,
                             permanent_water_available:bool=True,
                             mkdirs:bool=False) -> Tuple[str, str, str, str, Optional[str], str]:
    """
    For a given file (`tiff_file_name`) it returns the set of paths that the function generate_item produce.
    These paths are:
    - floodmap_path_dest. (.geojson)
    - gt_path (.tif)
    - meta_parent_path (.tif)
    - permanent_water_image_path_dest (.tif) or None if not permanent_water_available
    - s2_image_path_dest (.tif)
    Args:
        output_path: Path to produce the outputs
        file_name:
        permanent_water_available:
        mkdirs: make dirs if needed for the output paths
    Returns:
        s1_image_pre_path, s1_image_post_path, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, 
    """
    if permanent_water_available:
        permanent_water_image_path_dest = os.path.join(output_path, "PERMANENTWATERJRC", file_name+".tif").replace("\\", "/")
    else:
        permanent_water_image_path_dest = None

    output_path = str(output_path)
    s1_image_pre_path = os.path.join(output_path,"S1_pre",file_name+".tif").replace("\\", "/")
    s1_image_post_path = os.path.join(output_path,"S1_post",file_name+".tif").replace("\\", "/")
    meta_parent_path = os.path.join(output_path,"meta",file_name+".json").replace("\\", "/")


    floodmap_path_dest = os.path.join(output_path,"floodmaps",file_name+".geojson").replace("\\", "/")
    gt_path = os.path.join(output_path,"gt",file_name+".tif").replace("\\", "/")

    # makedir if not gs
    if mkdirs and not s1_image_post_path.startswith("gs"):
        fs = utils.get_filesystem(s1_image_post_path)
        for f in [s1_image_post_path, s1_image_pre_path, meta_parent_path, floodmap_path_dest, gt_path, permanent_water_image_path_dest]:
            if f is not None:
                fs.makedirs(os.path.dirname(f), exist_ok=True)

    return s1_image_pre_path, s1_image_post_path, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest


def pre_post_flood_image(metadatas2:pd.DataFrame, floodmap_date:datetime) -> Tuple[Any, datetime]:
    """
    Return dates pre and post to the floodmap_date

    Args:
        metadatas2:
        floodmap_date:

    Returns:

    """
    
    metadatas2 = metadatas2.loc[metadatas2.s2available]
    metadatas2['delay'] = metadatas2.datetime.apply(lambda x: (x - floodmap_date).total_seconds() / 3600.)
    metadatas2 = metadatas2.sort_values(by = 'delay',ascending=False)
    
    metadatas_pre = metadatas2.loc[metadatas2.delay < -10]
    metadatas_post = metadatas2.loc[metadatas2.delay > -10]
    
    index_pre= metadatas_pre.index[0]
    s1_date_pre = metadatas_pre.iloc[0].datetime
    index_post= metadatas_post.index[-1]
    s1_date_post = metadatas_post.iloc[-1].datetime

    return index_pre, index_post, s1_date_pre, s1_date_post

def gt_fun_v3(
    s1_image_pre_path: str,
    s1_image_post_path: str,
    floodmap: gpd.GeoDataFrame,
    metadata_floodmap: Dict,
    window: Optional[rasterio.windows.Window] = None,
    keep_streams: bool = False,
    permanent_water_image_path: Optional[str] = None,
    # cloudprob_image_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    New ground truth generating function for SAR segmentation
    Args:
    s1_image_post_path: str,
    s1_image_pre_path: str,
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
    gt = compute_water(
       s1_image_post_path,
       floodmap,
       window=window,
       permanent_water_path=permanent_water_image_path,
       keep_streams=keep_streams,
   )
    
    # Compute metadata of the ground truth
    metadata = metadata_floodmap.copy()
    
    # Compute stats of the water mask
    metadata["pixels flood water S2"] = int(np.sum(gt == 1))
    metadata["pixels hydro water S2"] = int(np.sum(gt == 2))
    metadata["pixels permanent water S2"] = int(np.sum(gt == 3))
    
    
    gt[gt==3] = 1 # set all water to 1 
    gt += 1
    # Compute stats of the GT
    metadata["pixels invalid"] = int(np.sum(gt == 0))
    metadata["pixels water"] = int(np.sum(gt == 2))
    metadata["pixels land"] = int(np.sum(gt == 1))

    metadata["gtversion"] = "v3"
    metadata["encoding_values"] = [
        {0: "invalid", 1: "land", 2: "water"},
    ]
    metadata["shape"] = list(gt.shape)
    # metadata["s2_image_path"] = os.path.basename(s2_image_path)
    metadata["permanent_water_image_path"] = (
        os.path.basename(permanent_water_image_path)
        if permanent_water_image_path is not None
        else "None"
    )
    # metadata["cloudprob_image_path"] = (
    #     os.path.basename(cloudprob_image_path)
    #     if cloudprob_image_path is not None
    #     else "None"
    # )
    # metadata["method clouds"] = "s2cloudless"

    with utils.rasterio_open_read(s1_image_post_path) as s2_src:
        metadata["bounds"] = s2_src.bounds
        metadata["crs"] = s2_src.crs
        metadata["transform"] = s2_src.transform

    return gt, metadata


def generate_item_v3(main_path:str, output_path:str, file_name:str,
                  overwrite:bool=False, pbar:Optional[tqdm.tqdm]=None,
                  gt_fun:Callable=None, delete_if_error:bool=True,
                  paths_function:Callable=None) -> bool:
    """
    Generates an "element" of the WorldFloods dataset with the expected naming convention. An "element" is a set of files
    that will be used for training the WorldFloods model. These files are copied in the `output_path` folowing the naming
    convention coded in `worldfloods_output_files` function.
     These files are:
    - shp vector floodmap. (subfolder floodmap)
    - tiff permanent water (subfolder PERMANENTWATERJRC)
    - tiff gt
    - tiff S1 pre and post
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
        if not overwrite:
            expected_outputs = worldfloods_v3_output_files(
                output_path, file_name, permanent_water_available=True, mkdirs=False)
            fsdest = utils.get_filesystem(expected_outputs[-1])

            must_process = False
            for e in expected_outputs:
                if e and not fsdest.exists(e):
                    must_process = True
                    break

            if not must_process:
                return True

        # Get input files and check that they all exist
        floodmap, permanent_water_path, meta_floodmap, s1_image_pre_path, s1_image_post_path = paths_function(main_path)

        # get output files
        s1_image_pre_path_dest, s1_image_post_path_dest, floodmap_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest = worldfloods_v3_output_files(
            output_path, file_name, permanent_water_available=permanent_water_path is not None, mkdirs=True)

    except Exception:
        warnings.warn(f"File {main_path} problem when computing input/output names")
        traceback.print_exc(file=sys.stdout)
        return False
    try:
        # generate gt, gt meta and copy all files to path_write
        fsdest = utils.get_filesystem(s1_image_pre_path_dest)

        if not fsdest.exists(gt_path_dest) or not fsdest.exists(meta_json_path_dest) or overwrite:
            if pbar is not None:
                pbar.set_description(f"Generating Ground Truth {file_name}...")

            # Copy s2_image_path to local before reading?
            # If so we will need also to copy cloudprob_path and permanent_water_path

            gt, gt_meta = gt_fun(
                s1_image_pre_path, 
                s1_image_post_path,
                floodmap,
                metadata_floodmap=meta_floodmap,
                keep_streams=True,
                permanent_water_image_path=permanent_water_path,  # Could be None!
            )

            if len(gt.shape) == 2:
                gt = gt[None]

            if pbar is not None:
                pbar.set_description(f"Saving GT {file_name}...")


            save_cog.save_cog(gt, gt_path_dest,
                              {"crs": gt_meta["crs"], "transform":gt_meta["transform"] ,"RESAMPLING": "NEAREST",
                               "compress": "lzw", "nodata": 0}, # In both gts 0 is nodata
                              descriptions= ["invalid/land/water/"],
                              tags=gt_meta)

            # upload meta json to bucket
            if pbar is not None:
                pbar.set_description(f"Saving meta {file_name}...")

            # save meta in local json file
            gt_meta["crs"] = str(gt_meta["crs"])
            gt_meta["transform"] = [gt_meta["transform"].a, gt_meta["transform"].b, gt_meta["transform"].c,
                                    gt_meta["transform"].d, gt_meta["transform"].e, gt_meta["transform"].f]

            utils.write_json_to_gcp(meta_json_path_dest, gt_meta)

        # Copy floodmap shapefiles
        if not fsdest.exists(floodmap_path_dest) or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving floodmap {file_name}...")

            utils.write_geojson_to_gcp(floodmap_path_dest, floodmap)                
        
        # Copy S2 image
        if not fsdest.exists(s1_image_pre_path_dest) or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving S1 image {file_name}...")
            
            _copy(s1_image_pre_path, s1_image_pre_path_dest, fs)
        if not fsdest.exists(s1_image_post_path_dest) or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving S1 image {file_name}...")
            
            _copy(s1_image_post_path, s1_image_post_path_dest, fs)
        
        
        # # Copy cloudprob
        # if cloudprob_path is not None and cloudprob_path_dest and (not fsdest.exists(cloudprob_path_dest) or overwrite):
        #     if pbar is not None:
        #         pbar.set_description(f"Saving cloud probs {file_name}...")
            
        #     _copy(cloudprob_path, cloudprob_path_dest, fs)
        
        # Copy permanent water
        if (permanent_water_image_path_dest is not None) and (not fsdest.exists(permanent_water_image_path_dest) or overwrite):
            if pbar is not None:
                pbar.set_description(f"Saving permanent water image {file_name}...")

            _copy(permanent_water_path, permanent_water_image_path_dest, fs)

    except Exception:
        warnings.warn(f"File input: {main_path} output S2 file: {s1_image_post_path} problem when computing Ground truth")
        traceback.print_exc(file=sys.stdout)

        if not delete_if_error:
            return False

        fsdest = utils.get_filesystem(s1_image_post_path_dest)
        files_to_delete = [s1_image_post_path_dest, s1_image_pre_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest,
                           floodmap_path_dest]
        for f in files_to_delete:
            if f and fsdest.exists(f):
                print(f"Deleting file {f}")
                fsdest.delete(f)

        return False

    return True

