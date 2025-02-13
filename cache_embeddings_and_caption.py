from nltk.corpus import stopwords
import captioning
import constants
import datasets_local
import fire
import logging
import model_wrappers
import nltk
import numpy as np
import open_clip
import os
import pandas as pd
import sys
import torch
import utils

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def cache_embeddings_and_caption_dataset(
    dataset="Waterbirds",
    clip: str = "openai/ViT-L-14",
    device: str = "cuda",
):
    """
    Captions dataset to create a vocabulary.
    Loads the initial and finetuned classification weights.
    Computes biases based on the weights and the word embeddings.
    """
    # Logging setup
    output_folder = os.path.join(constants.CACHE_PATH, "biases", dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device(device)

    clip_version, clip_architecture = clip.split("/")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_architecture, pretrained=clip_version
    )
    model.eval()
    model.to(device)
    print("CLIP temp", model.logit_scale.item())
    # dataset
    splits = ["train", "val", "test"]
    datasets = {
        split: datasets_local.ImageDataset(dataset, split=split, transform=preprocess)
        for split in splits
    }

    for split in splits:
        utils.cache_and_get_model_outputs(
            model=model.encode_image, dataset=datasets[split], device=device
        )

    # captioning dataset
    captioning.caption(dataset)


def cache_embeddings_civilcomments(device: str = "cuda"):
    comments_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR["CivilComments"],
        "civilcomments_coarse.csv",
    )
    df = pd.read_csv(comments_path)
    texts = df.comment_text.to_numpy()

    metadata_path = os.path.join(
        constants.DATA_PATH,
        constants.DATASET_DIR["CivilComments"],
        constants.METADATA_NAME["CivilComments"],
    )

    mtd = pd.read_csv(metadata_path)
    split_labels = mtd.split.to_numpy()

    # splits = ['train', 'val', 'test']
    splits = ["test", "val", "train"]
    device = torch.device(device)
    model = model_wrappers.SentenceEncoderWrapper()
    model.eval()
    model.to(device)

    output_dir = os.path.join(
        constants.CACHE_PATH,
        "model_outputs",
        "CivilComments",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in splits:
        output_path = os.path.join(output_dir, f"{split}.npy")
        # print('processing', split)
        split_comments = texts[split_labels == constants.DATASET_SPLITS[split]]
        split_embeddings = model.encode_texts_batched(
            split_comments, device, bs=128
        ).numpy()

        np.save(output_path, split_embeddings)


def cache_embeddings_(
    dataset="Waterbirds",
    clip: str = "openai/ViT-L-14",
    device: str = "cuda",
):
    if dataset == "CivilComments":
        cache_embeddings_civilcomments(device)
    else:
        cache_embeddings_and_caption_dataset(dataset, clip, device)


if __name__ == "__main__":
    fire.Fire(cache_embeddings_)
