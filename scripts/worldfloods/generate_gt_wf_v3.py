import tqdm
from ml4floods.data.utils import write_json_to_gcp
from pathlib import Path
from ml4floods.data.worldfloods.create_worldfloods_dataset import generate_item, worldfloods_output_files, worldfloods_extra_gcp_paths
from ml4floods.data.create_gt import generate_land_water_cloud_gt, generate_water_cloud_binary_gt
from ml4floods.data import utils
import os
import fsspec
import json
from typing import Callable, List
import pkg_resources

from ml4floods.data.s1_dataset_utils import  worldfloods_v3_paths, generate_item_v3, gt_fun_v3



def main(destination_path="", train_test_split_file:str = None, overwrite=False, prod_dev="0_DEV", cems_code="", aoi_code="",
          subset="all", dataset="extra"):
    
    staging_path = f"gs://ml4cc_data_lake/{prod_dev}/1_Staging/WorldFloods"
    if train_test_split_file is None:
        assert dataset in ["", "original", "extra"], f"Unexpected dataset {dataset}"
        
        train_test_split_file = pkg_resources.resource_filename("ml4floods",
                                                                f"data/configuration/train_test_split_{dataset}_dataset.json")
        
    fstraintest = utils.get_filesystem(train_test_split_file)  
    
    with fstraintest.open(train_test_split_file, "r") as fh:
        train_val_test_split = json.load(fh)

    
    cems_codes_test = set(s.split("_")[0] for s in train_val_test_split["test"])
    if "EMSR9284" in cems_codes_test:
        cems_codes_test.add("EMSR284")
    
    if subset != "all":
        train_val_test_split_new = {subset: train_val_test_split[subset]}
        train_val_test_split = train_val_test_split_new
    
    skip_unused = subset != "all"

    fs_ml4cc = fsspec.filesystem("gs", requester_pays=True)
    files_metadata_pickled = []

    
    for s in train_val_test_split:
        print(f"Subset {s} {len(train_val_test_split[s])} ids")
        # for f in train_val_test_split[s]:
        #     cems_code = f.split('_')[-4]
        #     aoi_code = f.split('_')[-3]
        #     files_metadata_pickled = [f"gs://{f}" for f in
        #                               fs_ml4cc.glob(f"{staging_path}/{cems_code}/{aoi_code}/flood_meta/*.pickle")]
    files_metadata_pickled = [f"gs://{f}" for f in
                              fs_ml4cc.glob(f"{staging_path}/*{cems_code}/*{aoi_code}/flood_meta/*.pickle")]
    print(f"Provided {len(files_metadata_pickled)} pickle files to generate the ML-ready dataset")
    

    # loop through files in the bucket
    problem_files = []
    count_files_per_split = {s:0 for s in train_val_test_split}
    with tqdm.tqdm(files_metadata_pickled, desc="Generating ground truth extra data") as pbar:
        for metadata_file in pbar:
            metadata_floodmap = utils.read_pickle_from_gcp(metadata_file)
            event_id = metadata_floodmap["event id"]

            # Find out which split to put the data in
            subset_iter = "unused"
            for split in train_val_test_split.keys():
            #     if (split != "test") and (metadata_floodmap["ems_code"] in cems_codes_test):
            #         subset_iter = "banned"

                if event_id in train_val_test_split[split]:
                    subset_iter = split
                    break

            # Do not process if subset is different
            if skip_unused and (subset_iter == subset):
                continue

            # Create destination folder if it doesn't exists
            path_write = os.path.join(destination_path, subset_iter).replace("\\", "/")
            if not path_write.startswith("gs:") and not os.path.exists(path_write):
                os.makedirs(path_write)
                
            status = generate_item_v3(metadata_file,
                                   path_write,
                                   file_name=event_id,
                                   overwrite=overwrite,
                                   pbar=pbar, gt_fun=gt_fun_v3,
                                   paths_function=worldfloods_v3_paths)
            if status:
                if subset_iter in count_files_per_split:
                    count_files_per_split[subset_iter]+= 1
            else:
                problem_files.append(metadata_file)

    for s in train_val_test_split:
        print(f"Split {s} expected {len(train_val_test_split[s])} copied {count_files_per_split[s]}")

    if len(problem_files) > 0:
        print("Files not generated that were expected:")
        for p in problem_files:
            print(p)


        
    pass


if __name__ == "__main__":
    import argparse

#     parser = argparse.ArgumentParser('Generate WorldFloods ML Dataset')
#     parser.add_argument('--version', default='v1_0', choices=["v1_0", "v2_0"],
#                         help="Which version of the ground truth we want to create (3-class) or multioutput binary")
#     parser.add_argument('--dataset', default='extra', choices=["", "original", "extra"],
#                         help="Use the old data '', the old data with the new pre-processing 'original' data or "
#                              "the newly downloaded data from Copernicus EMS with new pre-processing 'extra'")
#     parser.add_argument('--prod_dev', default='0_DEV', choices=["0_DEV", "2_PROD"],
#                         help="environment where the dataset would be created")
#     parser.add_argument('--overwrite', default=False, action='store_true',
#                         help="Overwrite the content in the folder {prod_dev}/2_Mart/worldfloods_{version}")
#     parser.add_argument('--cems_code', default="",
#                         help="CEMS Code to download images from. If empty string (default) download the images"
#                              "from all the codes")
#     parser.add_argument('--aoi_code', default="",
#                         help="CEMS AoI to download images from. If empty string (default) download the images"
#                              "from all the AoIs")
#     parser.add_argument('--subset', default="all",choices=["all", "train", "test", "val"],
#                         help="all/train/test/val subset to generate the ground truth only for those images"
#                              "Defaults to all")
#     parser.add_argument('--destination_path', default="",
#                         help="Destination path")

#     args = parser.parse_args()

#     main(version=args.version, overwrite=args.overwrite, prod_dev=args.prod_dev, dataset=args.dataset,
#          cems_code=args.cems_code, aoi_code=args.aoi_code,destination_parent_path=args.destination_path,
#          subset=args.subset)

    # main(destination_path='/home/kikeportales/erc/databases/SAR_Flood_Datasets/WORLDFLOODS/worldfloods_v3_0', train_test_split_file='/home/kikeportales/Documentos/Investigacio/Floods/notebooks/worldfloodsv3/split_wf_v3.json',
    #      cems_code='EMSR441',aoi_code = 'AOI07')
    main(destination_path='/home/kikeportales/erc/databases/SAR_Flood_Datasets/WORLDFLOODS/worldfloods_v3_0', train_test_split_file='/home/kikeportales/Documentos/Investigacio/Floods/notebooks/worldfloodsv3/split_wf_v3.json',
         cems_code= 'EMSR419', aoi_code='AOI01')

         
         
         