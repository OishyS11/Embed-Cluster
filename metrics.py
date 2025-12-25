# ----------------------------
# Metrics
# ----------------------------
def clustering_accuracy(y_true, y_pred, ignore_label: int | None = -1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ignore_label is not None:
        mask = (y_pred != ignore_label)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            return 0.0

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return float(cm[row_ind, col_ind].sum() / len(y_true))


def external_metrics(true, pred) -> dict:
    return {
        "Accuracy": clustering_accuracy(true, pred, ignore_label=-1),
        "NMI": normalized_mutual_info_score(true, pred),
        "ARI": adjusted_rand_score(true, pred),
        "FMI": fowlkes_mallows_score(true, pred),
    }


def internal_metrics(X, pred) -> dict:
    # Silhouette supports cosine; Davies-Bouldin is Euclidean-based
    # If HDBSCAN produced -1 noise, silhouette can be ill-defined; guard it.
    pred = np.asarray(pred)
    if len(np.unique(pred[pred != -1])) < 2:
        sil = np.nan
    else:
        sil = silhouette_score(X, pred, metric="cosine")
    try:
        dbi = davies_bouldin_score(X, pred)
    except Exception:
        dbi = np.nan
    return {"Silhouette(cosine)": sil, "Davies-Bouldin": dbi}