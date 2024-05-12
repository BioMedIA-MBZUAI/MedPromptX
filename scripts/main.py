import os
import logging
import sys
import ast
import re
import argparse
import datetime
from tqdm import tqdm
import wandb

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image


sys.path.append("..")
from src.utils import FlamingoProcessor
from main_utils import (
    clean_generation,
    eval_models,
    similarity_order_prompts,
    random_order_prompts,
)


def main():
    accelerator = Accelerator()  # when using cpu: cpu=True

    device = accelerator.device

    print("Loading model..")

    # >>> add your local path to LLM here:
    lang_encoder_path = args.lang_encoder
    # if not os.path.exists(lang_encoder_path):
    #     raise ValueError(
    #         "Llama model not yet set up, please check README for instructions!"
    #     )

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=args.visual_encoder,
        clip_vision_encoder_pretrained=args.visual_encoder_pretrained,
        lang_encoder_path=lang_encoder_path,
        tokenizer_path=lang_encoder_path,
        cross_attn_every_n_layers=4,
    )

    if args.model == "Med-Flamingo":
        # load Med-Flamingo checkpoint:
        if args.aws == "True":
            checkpoint_path = hf_hub_download("Med-Flamingo/Med-Flamingo", "model.pt")
        else:
            checkpoint_path = (
                "/l/users/mai.kassem/models/checkpoints/med-flamingo/model.pt"
            )
        print(f"Loaded Med-Flamingo checkpoint from {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )
    elif args.model == "OpenFlamingo":
        # load OpenFlamingo checkpoint:
        if args.aws == "True":
            checkpoint_path = hf_hub_download(
                "openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt"
            )
        else:
            checkpoint_path = "/l/users/mai.kassem/models/checkpoints/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
        print(f"Loaded OpenFlamingo checkpoint from {checkpoint_path}")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=False
        )

    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    model.eval()

    """
    Step 1: Load data
    """

    if args.aws == "True":
        root_dir = args.data_path
    else:
        # TODO: change this to your local path
        root_dir = f"/l/users/mai.kassem/datasets/MedPromptX/{args.data_path}"
    if args.prompt_type == "few-shot" and args.modality == "multimodal":
        prompts_df = pd.read_csv(f"{root_dir}/prompts_multimodal.csv")
        save_file = f"{args.dps_type}_{args.dps_modality}_few_multimodal_{args.model}"
    elif args.prompt_type == "few-shot" and args.modality == "image":
        prompts_df = pd.read_csv(f"{root_dir}/prompts_image_only.csv")
        save_file = f"{args.dps_type}_{args.dps_modality}_few_image_only_{args.model}"
    elif args.prompt_type == "zero-shot" and args.modality == "multimodal":
        prompts_df = pd.read_csv(f"{root_dir}/zero_prompts_multimodal.csv")
        save_file = f"zero_multimodal_{args.model}"
    elif args.prompt_type == "zero-shot" and args.modality == "image":
        prompts_df = pd.read_csv(f"{root_dir}/zero_prompts_image_only.csv")
        save_file = f"zero_image_only_{args.model}"

    # Add timestamp to save file
    now = datetime.datetime.now()
    save_file = f"{save_file}_{args.vg}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"

    # Create logger
    # =============
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler and set the log file path
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_file = f"logs/{save_file}.log"
    file_handler = logging.FileHandler(log_file)

    # Create formatter and add it to the file handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    wandb.init(project="MedPromptX", name=save_file)

    """
    Step 2: Define prompt 
    """
    if args.prompt_type == "few-shot":
        NUM_SHOTS = args.num_shots
    elif args.prompt_type == "zero-shot":
        NUM_SHOTS = 0

    if args.prompt_type == "few-shot":
        # example few-shot prompt:
        # ========================
        # original
        # prompt = "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. Follow the examples and answer the last question. <image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|><image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|><image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|><image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|><image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|><image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|><image>Question: Where is the largest mass located in the cerebellum? Answer:"

        if args.modality == "image":
            train_prompt = "You are a helpful medical assistant. You are being provided with images, a question, and an answer. Follow the examples and answer the last question. "

        elif args.modality == "multimodal":
            train_prompt = "You are a helpful medical assistant. You are being provided with images, a question given a set of patient laboratory test results, and an answer. Follow the examples and answer the last question. "

    elif args.prompt_type == "zero-shot":
        # example zero-shot prompt:
        # ========================
        if args.modality == "image":
            train_prompt = "You are a helpful medical assistant. You are being provided with an image and a question. Answer the question. "

        elif args.modality == "multimodal":
            train_prompt = "You are a helpful medical assistant. You are being provided with an image and a question given a set of patient laboratory test results. Answer the question. "

    logger.info(f"Prompt: {train_prompt}\n\n")
    logger.info(f"Prompt type: {args.prompt_type}\n")
    logger.info(f"Modality: {args.modality}\n")
    logger.info(f"Visual encoder: {args.visual_encoder}\n")
    logger.info(f"Visual encoder pretrained: {args.visual_encoder_pretrained}\n")
    logger.info(f"Language encoder: {args.lang_encoder}\n")
    logger.info(f"DPS: {args.dps}\n")
    logger.info(f"DPS type: {args.dps_type}\n")
    logger.info(f"DPS Modality: {args.dps_modality}\n")
    logger.info(f"DPS threshold: {args.dps_threshold}\n")
    logger.info(f"Visual Grounding: {args.vg}\n")
    logger.info(f"Data path: {args.data_path}\n\n")

    wandb.config.update(args)

    answers = []
    sum_prompts = 0
    for index, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
        if args.prompt_type == "few-shot" and args.dps == "True":
            image_paths_str = row["images"]
            image_paths_list = ast.literal_eval(image_paths_str)
            if args.vg == "True":
                demo_images = [
                    os.path.join(f"{root_dir}/prompts_images_grounded", p)
                    for p in image_paths_list
                ]
            else:
                demo_images = [
                    os.path.join(f"{root_dir}/prompts_images/", p)
                    for p in image_paths_list
                ]
            if args.dps_type == "similarity":
            # re-order few-shot samples based on similarity to test sample
                prompt, demo_images = similarity_order_prompts(
                    row["prompt"],
                    demo_images,
                    image_processor,
                    NUM_SHOTS,
                    args.dps_modality,
                    args.dps_threshold,
                )
                sum_prompts += len(demo_images)
            elif args.dps_type == "random":
                prompt, demo_images = random_order_prompts(
                    row["prompt"], demo_images, NUM_SHOTS
                )

        else:
            prompt = train_prompt + row["prompt"]

            image_paths_str = row["images"]
            image_paths_list = ast.literal_eval(image_paths_str)

            # Construct absolute paths for each image
            if args.vg == "True":
                demo_images = [
                    os.path.join(f"{root_dir}/prompts_images_grounded", p)
                    for p in image_paths_list
                ]
            else:
                demo_images = [
                    os.path.join(f"{root_dir}/prompts_images/", p)
                    for p in image_paths_list
                ]

        # Open each image using the Image module from the Pillow library
        demo_images = [Image.open(path) for path in demo_images]

        """
        Step 3: Preprocess data 
        """
        print("Preprocess data")
        pixels = processor.preprocess_images(demo_images)
        pixels = repeat(pixels, "N c h w -> b N T c h w", b=1, T=1)
        tokenized_data = processor.encode_text(prompt)

        """
        Step 4: Generate response 
        """

        # actually run few-shot prompt through model:
        print(f"Generate from {args.modality} {args.prompt_type} prompt")

        generated_text = model.generate(
            vision_x=pixels.to(device),
            lang_x=tokenized_data["input_ids"].to(device),
            attention_mask=tokenized_data["attention_mask"].to(device),
            max_new_tokens=10,
        )
        response = processor.tokenizer.decode(generated_text[0])
        response = clean_generation(response)

        print(f"{response}")

        if args.prompt_type == "few-shot":
            if len(re.findall("yes.", f"{response.lower()}")) > (NUM_SHOTS // 2):
                answer = "yes"
            elif len(re.findall("no.", f"{response.lower()}")) > (NUM_SHOTS // 2):
                answer = "no"
            else:
                answer = response

        elif args.prompt_type == "zero-shot":
            if len(re.findall("yes", f"{response.lower()}")) >= 1:
                answer = "yes"
            elif len(re.findall("no", f"{response.lower()}")) >= 1:
                answer = "no"
            else:
                answer = response

        answers.append(answer)
        torch.cuda.empty_cache()

    prompts_df["answer"] = answers
    prompts_df.to_csv(f"results/{save_file}.csv", index=False)
    report, precision, recall, f1_weighted, f1_micro, f1_macro, accuracy = eval_models(
        prompts_df["label"], answers
    )

    # Log precision, recall, f1, accuracy
    # ===================================
    logger.info(f"Report: {report}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 weighted: {f1_weighted}")
    logger.info(f"F1 micro: {f1_micro}")
    logger.info(f"F1 macro: {f1_macro}")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Average prompts: {sum_prompts // len(prompts_df)}")

    # Log precision, recall, f1, accuracy to wandb
    # ============================================
    wandb.log({"Precision": precision})
    wandb.log({"Recall": recall})
    wandb.log({"F1 weighted": f1_weighted})
    wandb.log({"F1 micro": f1_micro})
    wandb.log({"F1 macro": f1_macro})
    wandb.log({"Accuracy": accuracy})


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="MedPromptX Script")

    parser.add_argument(
        "--aws", type=str, default=False, help="Specify whether to use AWS"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["Med-Flamingo", "OpenFlamingo"],
        default="Med-Flamingo",
        help="Specify the model to use",
    )

    parser.add_argument(
        "--prompt_type",
        choices=["few-shot", "zero-shot"],
        default="few-shot",
        help="Specify the type of prompt (few-shot or zero-shot)",
    )

    # Add the arguments
    parser.add_argument(
        "--modality",
        type=str,
        choices=["image", "multimodal"],
        default="multimodal",
        help="Specify the modality (image or multimodal)",
    )
    parser.add_argument(
        "--visual_encoder",
        choices=["ViT-B-32", "ViT-L-14"],
        default="ViT-L-14",
        help="Specify the visual encoder (ViT-B-32 or ViT-L-14)",
    )
    parser.add_argument(
        "--visual_encoder_pretrained",
        choices=["openai", "imagenet"],
        default="openai",
        help="Specify the visual encoder pretrained on (openai or imagenet)",
    )
    parser.add_argument(
        "--lang_encoder",
        choices=[
            "/l/users/mai.kassem/models/llama-7b-hf",
            "huggyllama/llama-7b",
            "/l/users/mai.kassem/models/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base",
        ],
        default="/l/users/mai.kassem/models/llama-7b-hf",
        help="Specify the language encoder (llama or clinical-t5)",
    )
    parser.add_argument(
        "--num_shots",
        required=True,
        type=int,
        default=6,
        help="Specify the number of few-shot examples to use and change the data path accordingly",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        default="prompts_6_shot",
        help="Specify the root directory",
    )

    parser.add_argument(
        "--dps",
        type=str,
        default="True",
        help="Specify whether to change the order of examples or keep them the same",
    )
    parser.add_argument(
        "--dps_type",
        choices=["similarity", "random"],
        default="similarity",
        help="Specify the order type (similarity or random)",
    )
    parser.add_argument(
        "--dps_modality",
        default="both",
        choices=["text", "image", "both", "none"],
        help="Specify the data type for ordering (text, image, or both) only used if order_type=similarity",
    )
    parser.add_argument(
        "--dps_threshold",
        type=float,
        default=0.7,
        help="Specify the threshold for similarity ordering",
    )
    parser.add_argument(
        "--vg",
        type=str,
        default="True",
        help="Specify whether to use visual grounding",
    )
    # Parse the arguments
    args = parser.parse_args()
    print("Arguments:", args)

    main()

    wandb.finish()
