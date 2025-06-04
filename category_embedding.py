from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode category names
embeddings = model.encode(df_cat['categories'].tolist(), convert_to_tensor=True)

similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
similarity_df = pd.DataFrame(similarity_matrix, index=df_cat['categories'], columns=df_cat['categories'])


#Kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df_cat['cluster'] = kmeans.fit_predict(embeddings.cpu().numpy())

clustering = AgglomerativeClustering(n_clusters=10, metric='cosine', linkage='average')
df_cat['cluster'] = clustering.fit_predict(embeddings.cpu().numpy())


df_cat.to_csv("clustered_categories.csv", index=False)

