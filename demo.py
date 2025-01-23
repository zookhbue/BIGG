import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from model.diffGANmodel import DiffGANHDT, Discriminator
from dataset import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, train_config):
    print("testing ...")

    # Get dataset
    dataset = Dataset(
        "test.txt", args, sort=True, drop_last=True
    )
    batch_size = 1
    group_size = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
    )

    # Prepare model
    model = DiffGANHDT(args).to(device)
    print("generator configure: ", model)
    discriminator = Discriminator().to(device)
    print("discriminator configure: ", discriminator)
    ckpt_path = os.path.join(
        train_config["ckpt_path"],
        "trained_model.pth.tar"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["G"])

    for batchs in loader:
        input = [batchs['mel'], batchs['coarse']]
        output = model(input[0], input[1])

        A, cls_pred= output[1:3]
        pred_label = torch.softmax(cls_pred, dim=1)
        print("The generated BEC is ------> ", A)
        # print( A.shape, pred_label[0] )

        if pred_label[0][0] > 0.5:
            print("The predicted label is:  NC (probability={:.4f}). \n".format(pred_label[0][0]) )
        else:
            print("The predicted label is:  LMCI (probability={:.4f}). \n".format(pred_label[0][1]) )


        torchvision.utils.save_image(A[0], "output/example.jpg") # save the generated brain EC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        default="naive",
        help="training model type",
    )
    args = parser.parse_args()

    train_config={}
    train_config["ckpt_path"] = "ckpt/"

    # Log Configuration
    print("\n==================================== Testing Configuration ====================================")
    print(" ---> Type of Modeling:", args.model)
    print(" ---> Path of ckpt:", train_config["ckpt_path"])
    print("================================================================================================")

    main(args,train_config)
