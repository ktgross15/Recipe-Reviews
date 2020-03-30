# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

all_recipes_new_df = ... # Compute a Pandas dataframe to write into all_recipes_new


# Write recipe outputs
all_recipes_new = dataiku.Dataset("all_recipes_new")
all_recipes_new.write_with_schema(all_recipes_new_df)
