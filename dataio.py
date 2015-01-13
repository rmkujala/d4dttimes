import pandas as pd


def read_mobility_csv(fname, **kwargs):
    return pd.read_csv(fname,
                       names=["user_id", "timestamp", "site_id"],
                       **kwargs
                       )
