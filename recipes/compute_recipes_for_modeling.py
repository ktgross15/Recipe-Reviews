# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
recipes_prepared = dataiku.Dataset("recipes_prepared")
df = recipes_prepared.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['rating_40+'] = np.where(df['rating']>4.0, 1, 0)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['rating'] = df['rating'].apply(lambda x:round(x*2)/2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = df[df['num_reviews']>=3]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_bins(row):
    if row == np.NaN:
        return "Unknown"
    elif row < 100:
        return "<100"
    else:
        return ">=100"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numerical_cols = ['description_len','calories']

for col in numerical_cols:
    df[col] = df[col].apply(create_bins)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
recipes_for_modeling = dataiku.Dataset("recipes_for_modeling")
recipes_for_modeling.write_with_schema(df)