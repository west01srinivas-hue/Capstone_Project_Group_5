"""Issue clustering: multilingual sentence embeddings + k-means / HDBSCAN."""
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def kmeans_search(emb: np.ndarray, ks=range(6, 15), seed: int = 0):
    """Fit k-means for each k, return (best_labels, table of k / silhouette)."""
    rows, best = [], None
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)
        sil = silhouette_score(emb, labels)
        rows.append({"k": k, "silhouette": round(float(sil), 4)})
        if best is None or sil > best[0]:
            best = (sil, labels)
    return best[1], pd.DataFrame(rows)


def hdbscan_labels(emb: np.ndarray, min_cluster_size: int = 15, pca_dims: int = 20, seed: int = 0):
    """HDBSCAN on a PCA-reduced space (384-d embeddings are too sparse for density clustering)."""
    reduced = PCA(n_components=pca_dims, random_state=seed).fit_transform(emb)
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)


def summarize(df: pd.DataFrame, label_col: str, topic_col: str = "topic") -> pd.DataFrame:
    """Per cluster: size, languages, dominant Claude topic and its share (purity)."""
    rows = []
    for c, g in df.groupby(label_col):
        top = g[topic_col].value_counts()
        rows.append({
            "cluster": c, "size": len(g),
            "dominant_topic": top.index[0], "topic_purity": round(top.iloc[0] / len(g), 2),
            "languages": g.detected_language.value_counts().to_dict(),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
