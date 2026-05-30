import pandas as pd

#####################################################################
def make_temporal_train_test_split(df):
    """
    A temporal train/test split for CommunityFact.

    Test set:
      - stratified by domain x language
      - most recent examples in each group
      - approximately 500 examples for large groups
      - approximately 40% for smaller groups

    Also keeping all claims from the same tweetId-noteId pair in the same split.
    """

    df = df.copy()

    # Convert timestamp for sorting
    df["_time"] = pd.to_datetime(df["noteTimeStamp"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)

    # Keep tweet-note pairs together to avoid leakage
    df["_pair_id"] = df["tweetId"].astype(str) + "||" + df["noteId"].astype(str)

    test_pairs = set()

    for (domain, language), group in df.groupby(["domain", "language"]):
        n = len(group)

        # Simple test-size rule
        if n >= 1000:
            target_test_size = 500
        elif n >= 250:
            target_test_size = max(100, round(0.40 * n))
        else:
            target_test_size = max(1, round(0.30 * n))

        # Sort tweet-note pairs by most recent note timestamp
        pair_table = (
            group.groupby("_pair_id")
            .agg(
                latest_time=("_time", "max"),
                n_rows=("_pair_id", "size"),
            )
            .sort_values("latest_time", ascending=False)
        )

        # If there is only one pair in this group, keep it in train
        if len(pair_table) <= 1:
            continue

        selected_rows = 0

        # Leave at least the oldest pair in train
        for pair_id, row in pair_table.iloc[:-1].iterrows():
            test_pairs.add(pair_id)
            selected_rows += row["n_rows"]

            if selected_rows >= target_test_size:
                break

    df_test = df[df["_pair_id"].isin(test_pairs)].copy()
    df_train = df[~df["_pair_id"].isin(test_pairs)].copy()

    df_train = df_train.drop(columns=["_time", "_pair_id"]).reset_index(drop=True)
    df_test = df_test.drop(columns=["_time", "_pair_id"]).reset_index(drop=True)

    df_train = df_train.sample(frac=1, random_state=42)
    df_test = df_test.sample(frac=1, random_state=42)

    return df_train, df_test
#####################################################################

df = pd.read_csv("CF/cf_v1.csv")

print("\nBuilding train-test split...\n")
df_train, df_test = make_temporal_train_test_split(df)

df_train.to_csv("CF/cf_v1_train.csv", index=False)
df_test.to_csv("CF/cf_v1_test.csv", index=False)

print("\nYour benchmark dataset is ready to use. Go to data/CF folder.\n")