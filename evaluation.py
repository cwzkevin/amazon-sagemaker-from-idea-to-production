
import json
import os
import pathlib
import pickle as pkl
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":   
    
    # All paths are local for the processing container
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_x_path = "/opt/ml/processing/test/test_x.csv"
    test_y_path = "/opt/ml/processing/test/test_y.csv"
    output_dir = "/opt/ml/processing/evaluation"
    output_prediction_path = "/opt/ml/processing/output/"
        
    # Read model tar file
    with tarfile.open(model_path, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = xgb.Booster()
    model.load_model("xgboost-model")
    
    # Read test data
    X_test = xgb.DMatrix(pd.read_csv(test_x_path, header=None).values)
    y_test = pd.read_csv(test_y_path, header=None).to_numpy()

    # Run predictions
    probability = model.predict(X_test)

    # Evaluate predictions
    fpr, tpr, thresholds = roc_curve(y_test, probability)
    auc_score = auc(fpr, tpr)
    report_dict = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }

    # Save evaluation report
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
    
    # Save prediction baseline file - we need it later for the model quality monitoring
    pd.DataFrame({"prediction":np.array(np.round(probability), dtype=int),
                  "probability":probability,
                  "label":y_test.squeeze()}
                ).to_csv(os.path.join(output_prediction_path, 'prediction_baseline/prediction_baseline.csv'), index=False, header=True)
