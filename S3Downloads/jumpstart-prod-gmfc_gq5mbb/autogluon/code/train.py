import os
import json
import argparse
import yaml
import logging
from pprint import pprint

from autogluon.tabular import TabularDataset, TabularPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        logger.warn("More than one file is found in %s directory", path)
    logger.info("Using %s", file)
    filename = f"{path}/{file}"
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ---------------------------- Args parsing --------------------------------
    logger.info("Starting AutoGluon modelling")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN"))
    parser.add_argument('--target-column', type=str, default="binary_rating")
    parser.add_argument('--eval-metric', type=str, default="accuracy")
    parser.add_argument('--estimator', type=str, default="XGB")

    args, _ = parser.parse_known_args()
    logger.info("Args: %s", args)

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

    if args.n_gpus:
        logger.info(f"Running training job with the number of gpu: {int(args.n_gpus)}")

    # ----------------------------- Training -----------------------------------

    train_file = get_input_path(args.training_dir)
    train_data = TabularDataset(train_file)

    ag_predictor_args = {
        "label": args.target_column,
        "path": args.model_dir,
        "eval_metric": args.eval_metric,
        
    }
    
    hyperparameters = {args.estimator: {}} if args.estimator != "ensemble" else "default"
    
    predictor = TabularPredictor(**ag_predictor_args).fit(train_data, hyperparameters=hyperparameters)
    
    logger.info("Best model: %s", predictor.get_model_best())
    
    # Leaderboard
    lb = predictor.leaderboard()
    lb.to_csv(f'{args.output_data_dir}/leaderboard.csv', index=False)
    logger.info("Saved leaderboard to output.")
    logger.info("Training is completed.")
