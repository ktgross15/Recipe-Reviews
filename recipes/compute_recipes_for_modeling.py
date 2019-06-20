# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
recipes_prepared = dataiku.Dataset("recipes_prepared")
recipes_prepared_df = recipes_prepared.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

recipes_for_modeling_df = recipes_prepared_df # For this sample code, simply copy input to output


# Write recipe outputs
recipes_for_modeling = dataiku.Dataset("recipes_for_modeling")
recipes_for_modeling.write_with_schema(recipes_for_modeling_df)
