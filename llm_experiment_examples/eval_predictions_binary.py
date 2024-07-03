import json
from sklearn.metrics import *

import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description="Evaluate predicted output against gold annotations"
)

parser.add_argument(
    "--pred_path", required=True, help="Path to predicted txt file"
)

args = parser.parse_args()

logger.warning(f"args: {args}")

if args.pred_path:
    FILE_NAME = args.pred_path
    logger.warning(f"Reading predictions from {FILE_NAME}")
else:
    logger.error(f"Please provide predictions.txt path")

predictions = {}
with open(FILE_NAME, 'r') as fp:
  for line in fp.readlines():
      json_line = json.loads(line)
      predictions[json_line['id']] = json_line

pred, label = zip(*[(ele['binary'], ele['label']) for ele in predictions.values()])

assert len(pred)==len(label)

logger.warning(f"Evaluation completed")
logger.warning(f"Positive prediction: {sum([1 for ele in pred if int(ele)==1])}, Negative prediction: {sum([1 for ele in pred if int(ele)==0])}")
logger.warning(f"Pos/Total: {100*sum([1 for ele in pred if int(ele)==1])/len(pred):5f}%")
logger.info(f"f1: {f1_score(label, pred):6f}")
logger.warning(classification_report(label, pred, digits=4))
logger.warning(f"Prediction evaluated on {len(pred)} instances. Make sure that this number match with your original dataset!")