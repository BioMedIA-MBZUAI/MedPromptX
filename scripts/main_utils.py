import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from evaluate import load

bertscore = load("bertscore")

from PIL import Image

# for reproducibility or random prompts
np.random.seed(1) 

def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace("<unk> ", "").strip()


def eval_models(labels, answers):
    """Evaluate the model using sklearn metrics"""
    preds = []
    for i in range(len(answers)):
        pred = 1 if "yes" in answers[i] else 0
        preds.append(pred)

    report = classification_report(labels, preds)
    print(report)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 weighted: {f1_weighted}")
    print(f"F1 micro: {f1_micro}")
    print(f"F1 macro: {f1_macro}")
    print(f"Accuracy: {accuracy}")
    return report, precision, recall, f1_weighted, f1_micro, f1_macro, accuracy


def calc_cosine_similarity(a, b):
    """Calculate the cosine similarity between two arrays"""
    # Flatten the arrays to 1D vectors
    vector_A = a.flatten()
    vector_B = b.flatten()

    # Calculate the dot product of A and B
    dot_product = np.dot(vector_A, vector_B)

    # Calculate the Euclidean norms of A and B
    norm_A = np.linalg.norm(vector_A)
    norm_B = np.linalg.norm(vector_B)

    # Calculate the cosine similarity
    return dot_product / (norm_A * norm_B)

def order_images(images, image_processor, prompts, num_shots):
    """Order the images based on their similarity to the test sample"""
    train_images = images[:num_shots]
    test_images = images[num_shots:]
    train_prompts = prompts[:num_shots]
    test_prompts = prompts[num_shots:]

    scores = []
    for image, prompt in zip(train_images, train_prompts):
        image_embeddings = image_processor(Image.open(image))
        test_image_embeddings = image_processor(Image.open(test_images[0]))
        scores.append(calc_cosine_similarity(image_embeddings, test_image_embeddings))

    return train_images, test_images, scores


def order_clinical_data(prompt_texts, num_shots):
    """Order the temporal data based on their similarity to the test sample"""
    prompt_texts = prompt_texts.split("<|endofchunk|>")

    train_samples = prompt_texts[:num_shots]
    test_samples = prompt_texts[num_shots:]

    scores = []
    for sample in train_samples:
        scores.append(
            bertscore.compute(
                predictions=[sample], references=[test_samples[0]], lang="en"
            )["f1"][0]
        )

    return train_samples, test_samples, scores


def similarity_order_prompts(
    prompt, images, image_processor, num_shots, data_to_order="both", threshold=0.7
):
    """Sort prompts based on similarity to the first prompt or random order"""
    train_samples, test_samples, features_scores = order_clinical_data(
        prompt, num_shots
    )
    textual_prompts = train_samples.copy()
    textual_prompts.extend(test_samples)

    if images is not None and image_processor is not None:
        train_images, test_images, images_scores = order_images(
            images, image_processor, textual_prompts, num_shots
        )

    # Sort based on tabular data only/images only/both
    if data_to_order == "both":
        scores = np.mean([features_scores, images_scores], axis=0)
    elif data_to_order == "text":
        scores = features_scores.copy()
    elif data_to_order == "image":
        scores = images_scores.copy()

    # remove noisy samples by similarity threshold
    indices = [i for i, score in enumerate(scores) if score < threshold]
    train_samples = [
        sample for i, sample in enumerate(train_samples) if i not in indices
    ]
    if images is not None:
        train_images = [
            image for i, image in enumerate(train_images) if i not in indices
        ]

    # sort by ascending order (i.e. the most similar should be right before the test sample)
    train_samples = [x for _, x in sorted(zip(scores, train_samples))]
    prompts = train_samples.copy()
    prompts.extend(test_samples)

    if images is not None:
        train_images = [x for _, x in sorted(zip(scores, train_images))]
        prompt_images = train_images.copy()
        prompt_images.extend(test_images)
    else:
        prompt_images = None

    return "<|endofchunk|>".join(prompts), prompt_images


def random_order_prompts(prompt, images, num_shots):
    """Sort prompts based on random order"""
    prompt_texts = prompt.split("<|endofchunk|>")

    train_samples = prompt_texts[:num_shots]
    test_samples = prompt_texts[num_shots:]

    if images is not None:
        train_images = images[:num_shots]
        test_images = images[num_shots:]

    random_scores = np.random.rand(len(train_samples))

    train_samples = [x for _, x in sorted(zip(random_scores, train_samples))]
    prompts = train_samples.copy()
    prompts.extend(test_samples)

    if images is not None:
        train_images = [x for _, x in sorted(zip(random_scores, train_images))]
        prompt_images = train_images.copy()
        prompt_images.extend(test_images)
    else:
        prompt_images = None

    return "<|endofchunk|>".join(prompts), prompt_images
