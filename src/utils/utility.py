import clip


def get_all_class_features(model, class_names, device):
    """
    Computes the text embeddings for all classes in the dataset.
    Following image_ad55f7.png, labels are formatted as 'a photo of a [category]'.
    """
    # Create the prompts based on section 3.1
    prompts = [f"a photo of a {c}" for c in class_names]

    # Tokenize
    tokens = clip.tokenize(prompts).to(device)

    # Encode (using the student model so it learns)
    # We use the text_encoder (theta_T) as described in Eq. 2
    class_features = model.student.encode_text(tokens)

    # Normalize to get cosine similarity vectors
    class_features = class_features / class_features.norm(dim=-1, keepdim=True)

    return class_features
