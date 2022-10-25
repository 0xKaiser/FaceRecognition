import faiss
import torch
import clip
import os
import pandas as pd

# Read file KNN index 
#df = pd.read_parquet(".\data\embedding_folder\metadata\metadata_0.parquet")
#dung pandas de doc du lieu tu file dataset
df = pd.read_parquet(r'data\embedding_folder\metadata\metadata_0.parquet')

image_list = df["image_path"].tolist()

#ind = faiss.read_index(r".\data\knn.index")
ind = faiss.read_index(r'data\knn.index')

# Load the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load("ViT-B/32", device=device)

# Search image
def searchFace(image):
    image_tensor = preprocess(image)
    image_features = model.encode_image(torch.unsqueeze(image_tensor.to(device), dim=0))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_embeddings = image_features.cpu().detach().numpy().astype('float32')
    D, I = ind.search(image_embeddings, 5)
    if D[0][0] > 0.5: 
        name = os.path.basename(os.path.dirname(image_list[I[0][0]])) 
        print("Ten:", name, "voi do chinh xac:", D[0][0]*100)
        return name
    else:
        print('Ten: unknow - Do do chinh xac:', D[0][0]*100)
        return 'unknow'
