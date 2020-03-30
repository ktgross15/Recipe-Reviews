# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
gtom 



# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

epicurious_2_df = ... # Compute a Pandas dataframe to write into epicurious_2


# Write recipe outputs
epicurious_2 = dataiku.Dataset("epicurious_2")
epicurious_2.write_with_schema(epicurious_2_df)
