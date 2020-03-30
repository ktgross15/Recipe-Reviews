# from dataiku.customwebapp import *
import dataiku
import pandas as pd
from dataiku.apinode.predict.server import handle_predict
from flask import request
import json


SAMPLE_SIZE = 10000
THRESHOLD_CARDINALITY = 100

model_id = 's8ZCRTbJ'
# if model_id is None:
#     raise RuntimeError("Model not specified")
model = dataiku.Model(model_id)
predictor = model.get_predictor()

def get_categoricals(dataset, schema):
    """
    Detects low cardinality features and consider them as categoricals
    Returns the dataset schema enriched with the values of its categorical features
    """
    df = dataset.get_dataframe(limit=SAMPLE_SIZE)
    for column in schema:
        values = df[column['name']].unique()
        if len(values) < THRESHOLD_CARDINALITY:
            column['computedType'] = 'categorical'
            column['values'] = [None if pd.isnull(x) else x for x in values]
        else:
            column['computedType'] = column['type']
    return schema


@app.route('/get-dataset-schema')
def get_dataset_schema():
    dataset = dataiku.Dataset(request.args.get('dataset_name'))
    schema = dataset.read_schema()
    default_values = dataset.get_dataframe(limit=1)
    schema = get_categoricals(dataset, schema)
    return json.dumps({ "schema": schema, "defaultValues": default_values.to_json() })


@app.route('/score', methods=['POST'])
def score():
    payload = request.get_json()
    data = { "records": payload['records'], "schema": predictor.params.schema }
    prediction = handle_predict(predictor, data)
    return json.dumps({ "prediction": prediction })
