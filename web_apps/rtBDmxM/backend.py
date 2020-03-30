import dataiku
from dataiku.customwebapp import *
import pandas as pd
from flask import request


# Example:
# As the Python webapp backend is a Flask app, refer to the Flask
# documentation for more information about how to adapt this
# example to your needs.
# From JavaScript, you can access the defined endpoints using
# getWebAppBackendUrl('first_api_call')

model_id = get_webapp_config().get("input_model")
if model_id is None:
    raise RuntimeError("Model not specified")
model = dataiku.Model(model_id)
predictor = model.get_predictor()

@app.route('/first_api_call')
def first_call():
    max_rows = request.args.get('max_rows') if 'max_rows' in request.args else 500

    mydataset = dataiku.Dataset("REPLACE_WITH_YOUR_DATASET_NAME")
    mydataset_df = mydataset.get_dataframe(sampling='head', limit=max_rows)

    # Pandas dataFrames are not directly JSON serializable, use to_json()
    data = mydataset_df.to_json()
    return json.dumps({"status": "ok", "data": data})


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