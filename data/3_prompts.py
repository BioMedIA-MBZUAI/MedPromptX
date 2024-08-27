# %%
import pandas as pd
import os
import shutil
import random

img_root_path = "/physionet.org/files/mimic-cxr-jpg/2.0.0/"

SEED = 1

NUM_SHOTS = 6
NUM_SHOTS_PER_CLASS = NUM_SHOTS // 2

dest_root_path = f"prompts_{NUM_SHOTS}_shot"
dest_img_path = f"{dest_root_path}/prompts_images/"
if not os.path.exists(dest_img_path):
    os.makedirs(dest_img_path)

# %%

datasets = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

counts_labels = 0
for d in datasets:
    df = pd.read_csv("data_ehr_cxr/" + d + ".csv")
    df.drop_duplicates(subset=["study_id"], inplace=True)
    if len(df) <= 6:
        continue

    df = df.astype({"subject_id": "str"})
    df = df.astype({"hadm_id": "str"})
    df = df.astype({"stay_id": "str"})
    df = df.astype({"study_id": "str"})

    # Filter the dataframe where the label column is equal to 1
    positive_df = df[df[d] == 1]
    negative_df = df[df[d] == 0]
    if (
        len(positive_df) <= NUM_SHOTS_PER_CLASS
        or len(negative_df) <= NUM_SHOTS_PER_CLASS
    ):
        continue
    else:
        counts_labels += 1
    positive_df.reset_index(drop=True, inplace=True)
    negative_df.reset_index(drop=True, inplace=True)
    positive_df = positive_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    negative_df = negative_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # %%
    train_prompts_multimodal = ""
    train_prompts_image_only = ""
    positive_test_prompts = ""
    negative_test_prompts = ""
    question_first_part_multimodal = f"<image>Question: Is the patient likely to have {d}, given the following laboratory test results: "
    question_first_part_image_only = (
        f"<image>Question: Is the patient likely to have {d}"
    )
    question_last_part = "? "
    answer_positive = "Answer: yes."
    answer_negative = "Answer: no."
    answer_empty = "Answer:"
    endchunk = "<|endofchunk|>"  # This is a special token that indicates the end of a chunk (Flamingo models)

    # %%
    # Few-shot prompts
    save_file_path_multimodal = f"{dest_root_path}/prompts_multimodal.csv"
    save_file_path_image_only = f"{dest_root_path}/prompts_image_only.csv"
    prompts_dict_multimodal = {}
    prompts_dict_image_only = {}
    train_images = []
    for k in range(NUM_SHOTS_PER_CLASS):
        train_prompts_multimodal += (
            question_first_part_multimodal
            + positive_df["charts"][k]
            + question_last_part
            + answer_positive
            + endchunk
        )
        train_prompts_image_only += (
            question_first_part_image_only
            + question_last_part
            + answer_positive
            + endchunk
        )
        p_file_name = d + "_p" + str(k) + ".jpg"
        train_images.append(p_file_name)
        shutil.copy(
            img_root_path + positive_df["image_path"][k], dest_img_path + p_file_name
        )
    for k in range(NUM_SHOTS_PER_CLASS):
        train_prompts_multimodal += (
            question_first_part_multimodal
            + negative_df["charts"][k]
            + question_last_part
            + answer_negative
            + endchunk
        )
        train_prompts_image_only += (
            question_first_part_image_only
            + question_last_part
            + answer_negative
            + endchunk
        )
        n_file_name = d + "_n" + str(k) + ".jpg"
        train_images.append(n_file_name)
        shutil.copy(
            img_root_path + negative_df["image_path"][k], dest_img_path + n_file_name
        )

    # Positive examples
    for i in range(NUM_SHOTS_PER_CLASS, len(positive_df), 1):
        prompt_images = train_images.copy()
        test_prompt_multimodal = train_prompts_multimodal
        test_prompt_multimodal += (
            question_first_part_multimodal
            + positive_df["charts"][i]
            + question_last_part
            + answer_empty
        )
        test_prompt_image_only = train_prompts_image_only
        test_prompt_image_only += (
            question_first_part_image_only + question_last_part + answer_empty
        )

        file_name = d + "_p" + str(i) + "_test.jpg"
        prompt_images.append(file_name)

        shutil.copy(
            img_root_path + positive_df["image_path"][i],
            dest_img_path + file_name,
        )
        key = (
            str(i)
            + "_"
            + positive_df["subject_id"][i]
            + "_"
            + positive_df["hadm_id"][i]
            + "_"
            + positive_df["stay_id"][i]
        )
        prompts_dict_multimodal[key] = {
            "test_study_id": str(positive_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_multimodal,
            "label": 1,
        }
        prompts_dict_image_only[key] = {
            "test_study_id": str(positive_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_image_only,
            "label": 1,
        }

    # Negative examples
    for i in range(NUM_SHOTS_PER_CLASS, len(negative_df), 1):
        prompt_images = train_images.copy()
        test_prompt_multimodal = train_prompts_multimodal
        test_prompt_multimodal += (
            question_first_part_multimodal
            + negative_df["charts"][i]
            + question_last_part
            + answer_empty
        )
        test_prompt_image_only = train_prompts_image_only
        test_prompt_image_only += (
            question_first_part_image_only + question_last_part + answer_empty
        )

        file_name = d + "_n" + str(i) + "_test.jpg"
        prompt_images.append(file_name)
        shutil.copy(
            img_root_path + negative_df["image_path"][i],
            dest_img_path + file_name,
        )
        key = (
            str(i)
            + "_"
            + negative_df["subject_id"][i]
            + "_"
            + negative_df["hadm_id"][i]
            + "_"
            + negative_df["stay_id"][i]
        )
        prompts_dict_multimodal[key] = {
            "test_study_id": str(negative_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_multimodal,
            "label": 0,
        }
        prompts_dict_image_only[key] = {
            "test_study_id": str(negative_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_image_only,
            "label": 0,
        }

    # Save in CSV files
    df_prompts_multimodal = pd.DataFrame.from_dict(
        prompts_dict_multimodal, orient="index"
    )
    df_prompts_multimodal.to_csv(
        save_file_path_multimodal,
        mode="a",
        index=False,
        header=(not os.path.exists(save_file_path_multimodal)),
    )

    df_prompts_image_only = pd.DataFrame.from_dict(
        prompts_dict_image_only, orient="index"
    )
    df_prompts_image_only.to_csv(
        save_file_path_image_only,
        mode="a",
        index=False,
        header=(not os.path.exists(save_file_path_image_only)),
    )

    # %%
    # Zero-shot prompts
    save_file_path_multimodal = f"{dest_root_path}/zero_prompts_multimodal.csv"
    save_file_path_image_only = f"{dest_root_path}/zero_prompts_image_only.csv"
    prompts_dict_multimodal = {}
    prompts_dict_image_only = {}

    # Positive examples
    for i in range(NUM_SHOTS_PER_CLASS, len(positive_df), 1):
        prompt_images = []
        test_prompt_multimodal = (
            question_first_part_multimodal
            + positive_df["charts"][i]
            + question_last_part
            + answer_empty
        )

        test_prompt_image_only = (
            question_first_part_image_only + question_last_part + answer_empty
        )

        file_name = d + "_p" + str(i) + "_test.jpg"
        prompt_images.append(file_name)

        key = (
            str(i)
            + "_"
            + positive_df["subject_id"][i]
            + "_"
            + positive_df["hadm_id"][i]
            + "_"
            + positive_df["stay_id"][i]
        )
        prompts_dict_multimodal[key] = {
            "study_id": str(positive_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_multimodal,
            "label": 1,
        }
        prompts_dict_image_only[key] = {
            "study_id": str(positive_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_image_only,
            "label": 1,
        }

    # Negative examples
    for i in range(NUM_SHOTS_PER_CLASS, len(negative_df), 1):
        prompt_images = []
        test_prompt_multimodal = (
            question_first_part_multimodal
            + negative_df["charts"][i]
            + question_last_part
            + answer_empty
        )
        test_prompt_image_only = (
            question_first_part_image_only + question_last_part + answer_empty
        )

        file_name = d + "_n" + str(i) + "_test.jpg"
        prompt_images.append(file_name)

        key = (
            str(i)
            + "_"
            + negative_df["subject_id"][i]
            + "_"
            + negative_df["hadm_id"][i]
            + "_"
            + negative_df["stay_id"][i]
        )
        prompts_dict_multimodal[key] = {
            "study_id": str(negative_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_multimodal,
            "label": 0,
        }
        prompts_dict_image_only[key] = {
            "study_id": str(negative_df["study_id"][i]),
            "images": prompt_images,
            "prompt": test_prompt_image_only,
            "label": 0,
        }

    df_prompts_multimodal = pd.DataFrame.from_dict(
        prompts_dict_multimodal, orient="index"
    )
    df_prompts_multimodal.to_csv(
        save_file_path_multimodal,
        mode="a",
        index=False,
        header=(not os.path.exists(save_file_path_multimodal)),
    )

    df_prompts_image_only = pd.DataFrame.from_dict(
        prompts_dict_image_only, orient="index"
    )
    df_prompts_image_only.to_csv(
        save_file_path_image_only,
        mode="a",
        index=False,
        header=(not os.path.exists(save_file_path_image_only)),
    )

print(f"Number of labels with enough data: {counts_labels}")
