# pip install -q datasets
from datasets import load_dataset
import pandas as pd

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\S+@\S+", "", text)          # remove emails
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # keep letters/spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_20newsgroups_df_hf(subset="all", max_rows=0, max_per_class=0, seed=42):
    ds = load_dataset("SetFit/20_newsgroups")  # train / test splits :contentReference[oaicite:2]{index=2}

    if subset == "train":
        d = ds["train"]
    elif subset == "test":
        d = ds["test"]
    else:  # all
        d = pd.concat([pd.DataFrame(ds["train"]), pd.DataFrame(ds["test"])], ignore_index=True)
        # normalize below like the sklearn path expects
        df = d.rename(columns={"text": "text", "label": "label_int"})
        return _post_df(df, max_rows, max_per_class, seed)

    df = pd.DataFrame(d).rename(columns={"text": "text", "label": "label_int"})
    return _post_df(df, max_rows, max_per_class, seed)

def _post_df(df, max_rows, max_per_class, seed):
    # keep your existing cleaning if you want:
    df = df[df["text"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    if max_per_class and max_per_class > 0:
        df = (df.groupby("label_int", group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), max_per_class), random_state=seed))
                .reset_index(drop=True))

    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows].reset_index(drop=True)

    n_clusters = int(df["label_int"].nunique())
    # if you want string label names, you can just keep label_int as-is for metrics
    df["label"] = df["label_int"].astype(str)
    target_names = sorted(df["label_int"].unique().tolist())
    return df, target_names, n_clusters
