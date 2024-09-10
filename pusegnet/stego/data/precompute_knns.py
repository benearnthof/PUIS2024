from pusegnet.stego.data.data import ContrastiveSegDataset
from pusegnet.stego.modules import *
import os
from os.path import join
import hydra
import numpy as np
import torch.multiprocessing
import torch.multiprocessing
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
from tqdm import tqdm

# MXM EDIT BEGIN - Imports
from pusegnet.stego.mxm_utils import setup_wandb
# MXM EDIT END

def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(config_path="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/gitroot/stegorep/stegorep/src/configs", config_name="train_config.yaml")
def my_app(cfg: DictConfig) -> None:
    # setup_wandb(cfg)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(join(pytorch_data_dir, "nns"), exist_ok=True)

    pl.seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    image_sets = ["val", "train"]
    dataset_names = ["potsdam", "cityscapes"] #["cocostuff27", "cityscapes", "potsdam"]
    crop_types = ["five", None]

    # Uncomment these lines to run on custom datasets
    #dataset_names = ["directory"]
    #crop_types = [None]

    res = 224
    n_batches = 16
    if cfg.arch == "dino":
        from modules import DinoFeaturizer, LambdaLayer
        no_ap_model = torch.nn.Sequential(
            DinoFeaturizer(20, cfg),  # dim doesent matter
            LambdaLayer(lambda p: p[0]),
        ).cuda()
    elif cfg.arch == "dinov2":
        from modules import DinoV2Featurizer, LambdaLayer
        print("Using DINOv2 features for KNN")
        no_ap_model = torch.nn.Sequential(
            DinoV2Featurizer(20, cfg),  # dim does not matter
            LambdaLayer(lambda p: p[0]), # config is passed in use cfg.experiment name to distinguish between models
        ).cuda()
    else:
        cut_model = load_model(cfg.model_type, join(cfg.output_root, "data")).cuda()
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).cuda()
    par_model = torch.nn.DataParallel(no_ap_model)

    for crop_type in crop_types:
        for image_set in image_sets:
            for dataset_name in dataset_names:
                nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name

                print("Dataset: {}, Crop: {}, Image set: {}".format(nice_dataset_name, crop_type, image_set)) 
                feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
                    cfg.model_type, nice_dataset_name, image_set, crop_type, res))

                if not os.path.exists(feature_cache_file):
                    print("{} not found, computing".format(feature_cache_file))
                    dataset = ContrastiveSegDataset(
                        pytorch_data_dir=pytorch_data_dir,
                        dataset_name=dataset_name,
                        crop_type=crop_type,
                        image_set=image_set,
                        transform=get_transform(res, False, "center"),
                        target_transform=get_transform(res, True, "center"),
                        cfg=cfg,
                    )

                    loader = DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

                    with torch.no_grad():
                        normed_feats = get_feats(par_model, loader)
                        all_nns = []
                        step = normed_feats.shape[0] // n_batches
                        print(normed_feats.shape)
                        for i in tqdm(range(0, normed_feats.shape[0], step)):
                            torch.cuda.empty_cache()
                            batch_feats = normed_feats[i:i + step, :]
                            pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                            all_nns.append(torch.topk(pairwise_sims, 30)[1])
                            del pairwise_sims
                        nearest_neighbors = torch.cat(all_nns, dim=0)

                        np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                        print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
