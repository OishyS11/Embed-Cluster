# ----------------------------
# Embedding post-processing
# ----------------------------
def postprocess_embeddings(
    E: np.ndarray,
    center: bool = True,
    l2: bool = True,
    remove_top_pc: int = 0,
    pca_dim: int | None = None,
    whiten: bool = False,
) -> np.ndarray:
    X = E.astype(np.float32)

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    if remove_top_pc and remove_top_pc > 0:
        # common component removal (remove top principal directions)
        # (SVD on centered X)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        PCs = Vt[:remove_top_pc]         # [k,D]
        X = X - (X @ PCs.T) @ PCs        # subtract projection

    if pca_dim is not None:
        pca = PCA(n_components=int(pca_dim), whiten=bool(whiten), random_state=42)
        X = pca.fit_transform(X).astype(np.float32)

    if l2:
        X = normalize(X, norm="l2")

    return X
