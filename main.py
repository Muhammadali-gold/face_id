import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import streamlit as st
import numpy as np

# Load pretrained model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=0)

def get_embedding(img_path):
    # Load image
    img = Image.open(img_path).convert('RGB')
    # Detect face and crop
    face = mtcnn(img)
    if face is None:
        raise ValueError(f"No face detected in {img_path}")
    # Get embedding
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding

def has_multiple_face(img):
    mtcnn_x = MTCNN(keep_all=True,image_size=160, margin=0)  # keep_all=True ensures it returns multiple faces

    # Load image
    img = Image.open(img).convert('RGB')

    # Detect faces
    boxes, probs = mtcnn_x.detect(img)

    return boxes is not None and len(boxes) > 1

def cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

def get_similarity(img_path1, img_path2):
    # Example usage
    emb1 = get_embedding(img_path1)
    emb2 = get_embedding(img_path2)

    similarity = cosine_similarity(emb1, emb2)

    print("Face similarity:", similarity)

    return similarity


def main():
    st.title("Face Recognition App (Demo)")

    col1, col2 = st.columns(2)

    with col1:

        file_1 = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg", "jfif"],key='file_1')

        if file_1:
            st.image(file_1)

    with col2:

        file_2 = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg", "jfif"],key='file_2')

        if file_2:
            st.image(file_2)


    if st.button("Predict"):

        if not (file_1 and file_2):
            st.error("Please upload two image files")

        else:
            if has_multiple_face(file_1):
                st.error("First image has multiple faces.")
            elif has_multiple_face(file_2):
                st.error("Second image has multiple faces.")
            else:
                score = get_similarity(file_1, file_2)
                st.metric("Similarity", f"{score:.2f}")

if __name__ == '__main__':
    main()
