from ml4floods.data import ee_download, utils
from datetime import timedelta, datetime, timezone
import os
import pandas as pd
import warnings
import traceback
import sys


from typing import List, Callable, Optional
from georeader.readers import ee_query
import ee
from ml4floods.data.utils import get_filesystem, read_pickle_from_gcp
from ml4floods.data.ee_download import *
from ml4floods.data.ee_download import _check_all_downloaded
from shapely.geometry import box


def download_s1(area_of_interest: Polygon,
                   date_start_search: datetime, date_end_search: datetime,
                   path_bucket: str,
                   collection_name="S1", crs:str='EPSG:4326',
                   filter_fun:Callable[[pd.DataFrame], pd.Series]=None,
                   name_task:Optional[str]=None,
                   resolution_meters:float=10) -> List[ee.batch.Task]:
    """
    Download time series of S2 or Landsat images between search dates over the given area of interest. It saves the images
    on the path_bucket location. It only downloads images that satisfies the filter_s2 condition.

    Args:
        area_of_interest: shapely polygon with the AoI to download.
        date_start_search: start search date
        date_end_search: end search date
        path_bucket: path in the bucket to export the images. If the files in that bucket exists it does not download
        them.
        collection_name: "COPERNICUS/S2_HARMONIZED" for L1C Sentinel-2 images and ""COPERNICUS/S2_SR_HARMONIZED" for L2A images.
        crs: crs to export the images. To export them in utm based on location use the `convert_wgs_to_utm` function.
        filter_fun: function to filter the images to download. This function receives a dataframe with columns
            "cloud_probability", "valids" and "datetime" the output of this function should be boolean array of the
            with the number of rows of the dataframe that indicates which images of the dataframe to download.
        name_task: if not provided will use the basename of `path_bucket`
        force_s2cloudless:
        resolution_meters: resolution in meters to export the images

    Returns:
        List of running tasks and dataframe with metadata of the S2 files.

    """

    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    path_bucket_no_gs = path_bucket.replace("gs://", "")
    bucket_name = path_bucket_no_gs.split("/")[0]
    path_no_bucket_name = "/".join(path_bucket_no_gs.split("/")[1:])

    fs = get_filesystem("gs://")


    path_csv = os.path.join(path_bucket, "s1info.csv")

    if fs.exists(path_csv):
        data = process_metadata(path_csv, fs=fs)
        if _check_all_downloaded(data, date_start_search=date_start_search,
                                 date_end_search=date_end_search,
                                 filter_s2_fun=filter_fun,
                                 collection_name=collection_name):
            return []
        else:
            min_date = min(data["datetime"])
            max_date = max(data["datetime"])
            date_start_search = min(min_date, date_start_search)
            date_end_search = max(max_date, date_end_search)

    ee.Initialize()
    # area_of_interest_geojson = mapping(area_of_interest)
    # bounding_box_aoi = area_of_interest.bounds
    # bounding_box_pol = ee.Geometry(mapping(box(*bounding_box_aoi)))
    area_of_interest_geojson = mapping(area_of_interest)
    bounding_box_aoi = area_of_interest.bounds
    bounding_box_pol = ee.Geometry.Polygon(generate_polygon(bounding_box_aoi))

    img_col_info_local,img_col = ee_query.query_s1(area_of_interest,date_start_search, date_end_search, filter_duplicates=True,return_collection = True)

    if img_col is None:
        return []
    img_col_info_local = img_col_info_local.rename(columns={"utcdatetime": "datetime", "overlappercentage":"valids"}).drop(columns=["geometry"])
    img_col_info_local['valids'] = img_col_info_local['valids'].apply(lambda x: x/100)
    img_col_info_local["index_image_collection"] = np.arange(img_col_info_local.shape[0])

    # Get info of the S2 images (convert to table)
    # img_col_info_local = image_collection_fetch_metadata(img_col)

    n_images_col = img_col_info_local.shape[0]
    # Save S2 images as csv
    with fs.open(path_csv, "wb") as fh:
        img_col_info_local.to_csv(fh, index=False, mode="wb")

    print(f"Found {n_images_col} {collection_name} images between {date_start_search.isoformat()} and {date_end_search.isoformat()}")

    imgs_list = img_col.toList(n_images_col, 0)

    # TODO añadir aqui una funcion de georeader por si peta por tamaño?
    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        scale=resolution_meters,
        region = bounding_box_pol,
    )
    if filter_fun is not None:
        filter_good = filter_fun(img_col_info_local)

        if np.sum(filter_good) == 0:
            print("All images are bad")
            return []

        img_col_info_local_good = img_col_info_local[filter_good]
    else:
        img_col_info_local_good = img_col_info_local

    tasks = []
    for good_images in img_col_info_local_good.itertuples():
        img_export = ee.Image(imgs_list.get(good_images.index_image_collection))
        img_export = img_export.select(['VV','VH']).toFloat().clip(bounding_box_pol)

        date = good_images.datetime.strftime('%Y-%m-%d')

        if name_task is None:
            name_for_desc = os.path.basename(path_no_bucket_name)
        else:
            name_for_desc = name_task
        
        filename = os.path.join(path_no_bucket_name, date)
        desc = f"{name_for_desc}_{date}"
        task = mayberun(
            filename,
            desc,
            lambda: img_export,
            export_task_fun_img,
            overwrite=False,
            dry_run=False,
            bucket_name=bucket_name,
            verbose=2,
        )
        if task is not None:
            tasks.append(task)

    return tasks

def main(cems_code:str, aoi_code:str, threshold_invalids_before:float,
         threshold_invalids_after:float, days_before:int, days_after:int,
         only_one_previous:bool=False,
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
    # path_to_glob = f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*{aoi_code}/flood_meta/*.pickle"
    files_metatada_pickled = sorted([f"{prefix}{f}" for f in fs.glob(path_to_glob)])
    files_metatada_pickled.reverse()
    
    collection_names = ["S1"]
    resolutions_meters = [10]

    assert len(files_metatada_pickled) > 0, f"Not files found at {path_to_glob}"
    
    tasks = []
    for _i, meta_floodmap_filepath in enumerate(files_metatada_pickled):
        try:
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
                filter_before = (img_col_info_local["valids"] > (1 - threshold_invalids_before)) & \
                                (img_col_info_local["datetime"] < date_pre_flood) & \
                                (img_col_info_local["datetime"] >= date_start_search) & \
                                ~is_image_same_solar_day

                if only_one_previous and filter_before.any():
                    max_date = img_col_info_local.loc[filter_before, "datetime"].max()
                    filter_before &= (img_col_info_local["datetime"] == max_date)

                filter_after = (img_col_info_local["valids"] > (1 - threshold_invalids_after)) & \
                               (img_col_info_local["datetime"] <= date_end_search) & \
                               ((img_col_info_local["datetime"] >= satellite_date) | is_image_same_solar_day)
                return filter_before | filter_after
            
            
            tasks_iter = []
            basename_task = metadata_floodmap["ems_code"] + "_" + metadata_floodmap["aoi_code"]
            for collection_name_trigger, resolution_meters in zip(collection_names, resolutions_meters):
                folder_dest_s2 = os.path.join(folder_dest, collection_name_trigger)
                name_task = collection_name_trigger + "_" + basename_task
                tasks_iter.extend(download_s1(pol_scene_id,
                                                             date_start_search=date_start_search,
                                                             date_end_search=date_end_search,
                                                             crs=crs,
                                                             filter_fun=filter_images,
                                                             path_bucket=folder_dest_s2,
                                                             name_task=name_task,
                                                             resolution_meters=resolution_meters,
                                                             collection_name=collection_name_trigger))


            if len(tasks_iter) > 0:
                # Create csv and copy to bucket
                tasks.extend(tasks_iter)
            else:
                print(f"\tAll S1 data downloaded for product")
        except Exception:
            warnings.warn(f"Failed {meta_floodmap_filepath}")
            traceback.print_exc(file=sys.stdout)

    ee_download.wait_tasks(tasks)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Download Sentinel-1 GRD images for floodmaps in Staging')
    parser.add_argument('--cems_code', default="",
                        help="CEMS Code to download images from. If empty string (default) download the images"
                             "from all the codes")
    parser.add_argument('--aoi_code', default="",
                        help="CEMS AoI to download images from. If empty string (default) download the images"
                             "from all the AoIs")
    parser.add_argument('--only_one_previous', action='store_true')
    parser.add_argument("--metadatas_path", default="gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/",
                        help="gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/ for WorldFloods or "
                             "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/ for operational floods")
    parser.add_argument('--threshold_invalids_before', default=.3, type=float,
                        help="Threshold invalids before the event")
    parser.add_argument('--threshold_invalids_after', default=.70, type=float,
                        help="Threshold invalids after the event")
    parser.add_argument('--days_before', default=20, type=int,
                        help="Days to search after the event")
    parser.add_argument('--margin_pre_search', default=0, type=int,
                        help="Days to include as margin to search for pre-flood images")
    parser.add_argument('--days_after', default=20, type=int,
                        help="Days to search before the event")

    args = parser.parse_args()

    main(args.cems_code, aoi_code=args.aoi_code, threshold_invalids_before=args.threshold_invalids_before,
         threshold_invalids_after=args.threshold_invalids_after, days_before=args.days_before, metadatas_path=args.metadatas_path,
         only_one_previous=args.only_one_previous, 
         margin_pre_search=args.margin_pre_search,
         days_after=args.days_after)
