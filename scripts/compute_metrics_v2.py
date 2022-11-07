import numpy as np
import os

from ml4floods.models.config_setup import get_default_config
from ml4floods.models import dataset_setup
from ml4floods.models.model_setup import get_model, get_model_inference_function, get_channel_configuration_bands
from ml4floods.data.utils import write_json_to_gcp as save_json
from ml4floods.data.utils import get_filesystem
from ml4floods.models.utils.metrics import compute_metrics_v2
from typing import Tuple, Callable, List, Optional
import torch 

from ml4floods.models.utils.configuration import AttrDict
from ml4floods.data.worldfloods.configs import BANDS_S2, BANDS_L8
from ml4floods.data import create_gt
from ml4floods.models.postprocess import get_pred_mask_v2



def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code




def load_inference_function(model_path: str, device_name: str, max_tile_size: int = 1024,
                            th_water: float = .5,
                            th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
                            collection_name:str="S2",
                            distinguish_flood_traces:bool=False) -> Tuple[
    Callable[[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]], AttrDict]:
    if model_path.endswith("/"):
        experiment_name = os.path.basename(model_path[:-1])
        model_folder = os.path.dirname(model_path[:-1])
    else:
        experiment_name = os.path.basename(model_path)
        model_folder = os.path.dirname(model_path[:-1])

    config_fp = os.path.join(model_path, "config.json").replace("\\", "/")
    config = get_default_config(config_fp)

    # The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory constrained environment set this value to 128
    config["model_params"]["max_tile_size"] = max_tile_size

    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)
    model.to(device_name)
    inference_function = get_model_inference_function(model, config, apply_normalization=True,
                                                      activation=None)

    if config.model_params.get("model_version", "v1") == "v2":

        channels = get_channel_configuration_bands(config.data_params.channel_configuration,
                                                   collection_name=collection_name)
        if distinguish_flood_traces:
            if collection_name == "S2":
                band_names_current_image = [BANDS_S2[iband] for iband in channels]
                mndwi_indexes_current_image = [band_names_current_image.index(b) for b in ["B3", "B11"]]
            elif collection_name == "Landsat":
                band_names_current_image = [BANDS_L8[iband] for iband in channels]
                # TODO ->  if not all(b in band_names_current_image for b in ["B3","B6"])
                mndwi_indexes_current_image = [band_names_current_image.index(b) for b in ["B3", "B6"]]

        # Add post-processing of binary mask
        def predict(s2l89tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                s2l89tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: think cloud, 4: flood trace}
            """
            with torch.no_grad():
                pred = inference_function(s2l89tensor)[0]  # (2, H, W)
                

                if distinguish_flood_traces:
                    s2l89mndwibands = s2l89tensor.squeeze(0)[mndwi_indexes_current_image, ...].float()

                    # Green − SWIR1)/(Green + SWIR1)
                    mndwi = (s2l89mndwibands[0] - s2l89mndwibands[1]) / (s2l89mndwibands[0] + s2l89mndwibands[1] + 1e-6)

                    return pred, mndwi
                else:
                    return pred

    else:
        def predict(s2l89tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                s2l89tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: cloud, 4: flood trace}
            """
            with torch.no_grad():
                pred = inference_function(s2l89tensor)[0]  # (3, H, W)
                

                if distinguish_flood_traces:
                    s2l89mndwibands = s2l89tensor.squeeze(0)[mndwi_indexes_current_image, ...].float()
                    # Green − SWIR1)/(Green + SWIR1)
                    mndwi = (s2l89mndwibands[0] - s2l89mndwibands[1]) / (s2l89mndwibands[0] + s2l89mndwibands[1] + 1e-6)
                    
                    return pred, mndwi
                else:
                    return pred

    return predict, config


def main(experiment_path:str, path_to_splits=None, overwrite=False, device:Optional[torch.device]=None,
         max_tile_size:int=1_024, distinguish_flood_traces = False):
    """
    Compute metrics of a given experiment and saves it on the experiment_path folder

    Args:
        experiment_path: /path/to/folder/with/config.json file
        path_to_splits: e.g. /worldfloods/worldfloods_v1_0/
        overwrite:
        device:
        max_tile_size: Size to tile the GeoTIFFs
    """

    config_fp = os.path.join(experiment_path, "config.json").replace("\\","/")
    config = get_default_config(config_fp)

    inference_function, channels = load_inference_function(experiment_path, device, max_tile_size=max_tile_size, 
                                                           distinguish_flood_traces=distinguish_flood_traces)

    if path_to_splits is not None:
        config.data_params.path_to_splits = path_to_splits  # local folder where data is located

    config.data_params.train_test_split_file = ""
    if "filter_windows" in config["data_params"]:
        del config["data_params"]["filter_windows"]
    
    ### METRICS COMPUTATION #### 
    data_module = dataset_setup.get_dataset(config["data_params"])
    # # data_module.filenames_train_test['test'] = {'gt':[ '/home/kikeportales/erc/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/test/gt/EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1.tif'], 
    # #                                             'S2': [ '/home/kikeportales/erc/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/test/S2/EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1.tif']}
    # # data_module.test_files =  [ '/home/kikeportales/erc/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/test/S2/EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1.tif']
    # # data_module.test_dataloader.dataset.image_files = [ '/home/kikeportales/erc/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/test/S2/EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1.tif']
    # data_module.test_dataloader().dataset.image_files = [ '/home/kikeportales/erc/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/test/S2/EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1.tif']
    
    
    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:
    # for dl, dl_name in [ (data_module.val_dataloader(), "val")]:        
        metrics_file = os.path.join(experiment_path, f"{dl_name}.json").replace("\\","/")
        metrics_file = '/home/kikeportales/Escritorio/prova.json' # DELETE ME 
        fs = get_filesystem(metrics_file)
        if not overwrite and fs.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue

        mets = compute_metrics_v2(
            dl,
            inference_function, threshold_water=0.5,
            plot=False,
            mask_clouds=True,
            distinguish_flood_traces=distinguish_flood_traces)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        save_json(metrics_file, mets)
        
        
if __name__ == '__main__':
    from tqdm import tqdm
    import traceback
    import sys
    import argparse

    parser = argparse.ArgumentParser('Run metrics on test and val subsets for the provided models')
    parser.add_argument("--experiment_path", default="",
                        help="""
                        Path with config.json and model.pt files to load the model.
                        If not provided it will glob the --experiment_folder to compute metrics of all models
                        """)
    parser.add_argument("--experiment_folder", default="",help="""
                        Folder with folders with models. Each of the model folder is expected to have a config.json and 
                        model.pt files to load the model.
                        If --experiment_path provided will ignore this argument 
                        """)
    parser.add_argument("--max_tile_size", help="Size to tile the GeoTIFFs", type=int, default=1_024)
    parser.add_argument('--distinguish_flood_traces', default=False, action='store_true',
                        help="Use MNDWI to distinguish flood traces")
    parser.add_argument("--path_to_splits", required=True, help="path to test and val folders")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device)

    if args.experiment_path == "":
        # glob experiment folder
        fs = get_filesystem(args.experiment_folder)
        if args.experiment_folder.startswith("gs"):
            prefix = "gs://"
        else:
            prefix = ""
        experiment_paths = [f"{prefix}{os.path.dirname(f)}" for f in fs.glob(os.path.join(args.experiment_folder,"*","config.json").replace("\\","/"))]
        assert len(experiment_paths) > 0, "No models found in "+os.path.join(args.experiment_folder,"*","config.json").replace("\\","/")
    else:
        experiment_paths = [args.experiment_path]

    for ep in tqdm(experiment_paths):
        try:
            main(experiment_path=ep, path_to_splits=args.path_to_splits, device=device)
        except Exception:
            print(f"Error in experiment {ep}")
            traceback.print_exc(file=sys.stdout)
    # main(experiment_path='/home/kikeportales/erc/home/Projectes/ml4floods/2_MLModelMart/WF2_unet_full_norm/', path_to_splits = '/home/kikeportales/erc/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/',
    #      device = 'cpu', distinguish_flood_traces = True)

