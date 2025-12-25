# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # 20 Newsgroups controls
    ap.add_argument("--subset", type=str, default="all", choices=["train", "test", "all"])
    ap.add_argument("--keep_headers", action="store_true", help="If set, DO NOT remove headers/footers/quotes.")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = use all rows")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = no per-class cap")

    # Embedding model
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model_type", type=str, default="decoder-only",
                    choices=["encoder-only", "decoder-only", "encoder-decoder"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls", "last"])
    ap.add_argument("--trust_remote_code", action="store_true")

    # Post-processing
    ap.add_argument("--pp_center", action="store_true")
    ap.add_argument("--pp_no_center", action="store_true")
    ap.add_argument("--pp_l2", action="store_true")
    ap.add_argument("--pp_no_l2", action="store_true")
    ap.add_argument("--pp_remove_top_pc", type=int, default=1)
    ap.add_argument("--pp_pca_dim", type=int, default=256)
    ap.add_argument("--pp_whiten", action="store_true")

    # Clustering
    ap.add_argument("--hdb_min_cluster_size", type=int, default=20)
    ap.add_argument("--run_spectral", action="store_true")

    ap.add_argument("--out_csv", type=str, default="20news_Qwen-embeddings_clustering.csv")

    # NOTE: in notebooks, keep parse_args([]). In a real script, change to parse_args().
    args = ap.parse_args([])

    # Defaults: center + l2 unless explicitly disabled
    center = True
    l2 = True
    if args.pp_center:
        center = True
    if args.pp_no_center:
        center = False
    if args.pp_l2:
        l2 = True
    if args.pp_no_l2:
        l2 = False

    # Load 20 Newsgroups
    df, target_names, n_clusters = load_20newsgroups_df_hf(subset=args.subset, max_rows=args.max_rows, max_per_class=args.max_per_class)

    print("Loaded 20 Newsgroups:", df.shape, "| n_clusters:", n_clusters)

    texts = df["clean_text"].tolist()

    # Embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    emb_t = get_embeddings(
        model_name=args.model_name,
        model_type=args.model_type,
        texts=texts,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        trust_remote_code=args.trust_remote_code,
    )
    print("Embeddings tensor:", emb_t.shape)

    E = emb_t.numpy()
    E_pp = postprocess_embeddings(
        E,
        center=center,
        l2=l2,
        remove_top_pc=max(0, int(args.pp_remove_top_pc)),
        pca_dim=None if args.pp_pca_dim <= 0 else int(args.pp_pca_dim),
        whiten=bool(args.pp_whiten),
    )
    print("Post-processed embeddings:", E_pp.shape)

    # For cosine-friendly clustering
    E_norm = normalize(E_pp, norm="l2")

    # Clustering
    results = {}

    # A) KMeans (Euclidean)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(E_pp)
    results["KMeans (Euclidean)"] = {
        **external_metrics(df["label_int"], kmeans_labels),
        **internal_metrics(E_pp, kmeans_labels),
    }

    # B) Spherical KMeans
    spherical = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    spherical_labels = spherical.fit_predict(E_norm)
    results["Spherical KMeans"] = {
        **external_metrics(df["label_int"], spherical_labels),
        **internal_metrics(E_norm, spherical_labels),
    }

    # C) Agglomerative (cosine)
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    agg_labels = agg.fit_predict(E_pp)
    results["Agglomerative (cosine)"] = {
        **external_metrics(df["label_int"], agg_labels),
        **internal_metrics(E_pp, agg_labels),
    }

    # D) HDBSCAN (noise-aware metrics)
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=int(args.hdb_min_cluster_size),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    hdb_labels = hdb.fit_predict(E_norm)
    results["HDBSCAN"] = {
        **external_metrics(df["label_int"], hdb_labels), # Removed ignore_label here
        **internal_metrics(E_norm, hdb_labels),
    }

    # E) Spectral (optional; can be slow)

    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=42,
    )
    spectral_labels = spectral.fit_predict(E_norm)
    results["Spectral (kNN)"] = {
        **external_metrics(df["label_int"], spectral_labels),
        **internal_metrics(E_norm, spectral_labels),
    }

    # Save
    results_df = pd.DataFrame(results).T
    print("\n===== Results =====")
    print(results_df)
    results_df.to_csv(args.out_csv, index=True)
    print("\nSaved:", args.out_csv)


if __name__ == "__main__":
    main()