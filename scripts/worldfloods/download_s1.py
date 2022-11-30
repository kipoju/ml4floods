
from ml4floods.data import ee_download, utils
from datetime import timedelta, datetime, timezone
import os
import pandas as pd
import warnings
import traceback
import sys



def main(cems_code:str, aoi_code:str, threshold_invalids_before:float=0.8,
         threshold_invalids_after:float=0.8, days_before:int=20, days_after:int=20,
         collection_placeholder:str = "S1_GRD", only_one_previous:bool=False,
         margin_pre_search:int=0,
         metadatas_path:str="gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/"):
    """

    Args:
        cems_code:
        aoi_code:
        threshold_clouds_before:
        threshold_clouds_after:
        threshold_invalids_before:
        threshold_invalids_after:
        days_before:
        days_after:
        collection_placeholder: S2, Landsat or both
        only_one_previous:
        margin_pre_search:
        force_s2cloudless:
        metadatas_path:

    Returns:

    """

    fs = utils.get_filesystem(metadatas_path)
    path_to_glob = os.path.join(metadatas_path,f"{cems_code}*",f"{aoi_code}*", "flood_meta", "*.pickle").replace("\\", "/")
    prefix = "gs://" if metadatas_path.startswith("gs") else ""
    
    files_metatada_pickled = sorted([f"{prefix}{f}" for f in fs.glob(path_to_glob)])
    files_metatada_pickled.reverse()
    
    assert len(files_metatada_pickled) > 0, f"Not files found at {path_to_glob}"
    
    # Set collections to download
    if collection_placeholder == "S1_GRD":
        collection_name = ["S1_GRD"]
        resolution_meters = [10]

    tasks = []
    
    for _i, meta_floodmap_filepath in enumerate(files_metatada_pickled):
    
        metadata_floodmap = utils.read_pickle_from_gcp(meta_floodmap_filepath)
        pol_scene_id = metadata_floodmap["area_of_interest_polygon"]
        satellite_date = datetime.strptime(metadata_floodmap["satellite date"].strftime("%Y-%m-%d %H:%M:%S"),
                                           "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    
        folder_dest = os.path.dirname(os.path.dirname(meta_floodmap_filepath))
    
        # Compute arguments to download the images
    
        date_start_search = satellite_date + timedelta(days=-days_before)
        date_end_search = min(datetime.today().astimezone(timezone.utc),
                              satellite_date + timedelta(days=days_after))
        print(f"{_i + 1}/{len(files_metatada_pickled)} processing {meta_floodmap_filepath} between {date_start_search.strftime('%Y-%m-%d')} and {date_end_search.strftime('%Y-%m-%d')}")
    
        # Set the crs to UTM of the center polygon
        lon, lat = list(pol_scene_id.centroid.coords)[0]
        crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)
    
        date_pre_flood = satellite_date - timedelta(days=margin_pre_search)
        
        def filter_images(img_col_info_local:pd.DataFrame) -> pd.Series:
            is_image_same_solar_day = img_col_info_local["datetime"].apply(lambda x: (satellite_date - x).total_seconds() / 3600. < 10)
            filter_before = ((img_col_info_local["overlap"] > (threshold_invalids_before)) & \
                            (img_col_info_local["datetime"] < date_pre_flood) & \
                            (img_col_info_local["datetime"] >= date_start_search) & \
                            ~is_image_same_solar_day)

            if only_one_previous and filter_before.any():
                max_date = img_col_info_local.loc[filter_before, "datetime"].max()
                filter_before &= (img_col_info_local["datetime"] == max_date)

            filter_after = (img_col_info_local["overlap"] > (threshold_invalids_after)) & \
                           (img_col_info_local["datetime"] <= date_end_search) & \
                           ((img_col_info_local["datetime"] >= satellite_date) | is_image_same_solar_day)
            return filter_before | filter_after

        
        tasks_iter = []
        basename_task = metadata_floodmap["ems_code"] + "_" + metadata_floodmap["aoi_code"]
        for collection_name_trigger, resolution in zip(collection_name, resolution_meters):

            folder_dest_s2 = os.path.join(folder_dest, collection_name_trigger)
            name_task = collection_name_trigger + "_" + basename_task
            tasks_iter.extend(ee_download.download_s1(pol_scene_id,
                                                         date_start_search=date_start_search,
                                                         date_end_search=date_end_search,
                                                         crs=crs,
                                                         filter_fun=filter_images,
                                                         path_bucket=folder_dest_s2,
                                                         name_task=name_task,
                                                         resolution_meters=resolution,
                                                         collection_name=collection_name_trigger))
    
    
        if len(tasks_iter) > 0:
            # Create csv and copy to bucket
            tasks.extend(tasks_iter)
        else:
            print(f"\tAll S1 data downloaded for product")
    
        # download permanent water
        folder_dest_permament = os.path.join(folder_dest, "PERMANENTWATERJRC")
        task_permanent = ee_download.download_permanent_water(pol_scene_id, date_search=satellite_date,
                                                              path_bucket=folder_dest_permament,
                                                              name_task="PERMANENTWATERJRC"+basename_task,
                                                              crs=crs)
        if task_permanent is not None:
            tasks.append(task_permanent)

        ee_download.wait_tasks(tasks)
        

if __name__ == '__main__':
    import argparse

    # parser = argparse.ArgumentParser('Download Sentinel-2 and Landsat-8/9 images for floodmaps in Staging')
    # parser.add_argument('--cems_code', default="",
    #                     help="CEMS Code to download images from. If empty string (default) download the images"
    #                           "from all the codes")
    # parser.add_argument('--aoi_code', default="",
    #                     help="CEMS AoI to download images from. If empty string (default) download the images"
    #                           "from all the AoIs")
    # parser.add_argument('--only_one_previous', action='store_true')
    # parser.add_argument('--noforce_s2cloudless', action='store_true')
    # parser.add_argument("--collection_name", choices=["Landsat", "S2", "both"], default="S2")
    # parser.add_argument("--metadatas_path", default="gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/",
    #                     help="gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/ for WorldFloods or "
    #                           "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/ for operational floods")
    # parser.add_argument('--threshold_clouds_before', default=.3, type=float,
    #                     help="Threshold clouds before the event")
    # parser.add_argument('--threshold_clouds_after', default=.95, type=float,
    #                     help="Threshold clouds after the event")
    # parser.add_argument('--threshold_invalids_before', default=.3, type=float,
    #                     help="Threshold invalids before the event")
    # parser.add_argument('--threshold_invalids_after', default=.70, type=float,
    #                     help="Threshold invalids after the event")
    # parser.add_argument('--days_before', default=20, type=int,
    #                     help="Days to search after the event")
    # parser.add_argument('--margin_pre_search', default=0, type=int,
    #                     help="Days to include as margin to search for pre-flood images")
    # parser.add_argument('--days_after', default=20, type=int,
    #                     help="Days to search before the event")

    # args = parser.parse_args()

    # main(args.cems_code, aoi_code=args.aoi_code, threshold_clouds_before=args.threshold_clouds_before,
    #       threshold_clouds_after=args.threshold_clouds_after, threshold_invalids_before=args.threshold_invalids_before,
    #       threshold_invalids_after=args.threshold_invalids_after, days_before=args.days_before,
    #       collection_placeholder=args.collection_name, metadatas_path=args.metadatas_path,
    #       only_one_previous=args.only_one_previous, force_s2cloudless=not args.noforce_s2cloudless,
    #       margin_pre_search=args.margin_pre_search,
    #       days_after=args.days_after)
    
    main(cems_code='EMSR419', aoi_code='AOI01', days_after=10, days_before=5, threshold_invalids_before = 0.8)
    # # main(cems_code='EMSR342', aoi_code='04JULIACREEK', days_after=30, days_before=30)
    # # main(cems_code='EMSR482', aoi_code='AOI02', days_after=20, days_before=20)
    

