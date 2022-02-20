# %%

import pandas as pd

# %%

full_df = pd.read_csv("train/iris.csv")

# %%

sample = full_df.groupby("class").apply(lambda x: x.sample(n=10))

# %%

sample.drop(columns=["class"], inplace=True)

# %%

sample.to_json("test_batch_inference_input.jsonlines", orient="records", lines=True)

# %%
