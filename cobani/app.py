import argparse
import configparser

import lib.train
import lib.nest

if __name__ == "__main__":
  desc = "Cobani security. " \
         "Allows to train a model with the downloaded data from Raspberry Pi and Nest cameras."
  parser = argparse.ArgumentParser(description=desc)

  config = configparser.ConfigParser()
  config.read('.cobani')

  # Train machine learning model and store it locally at directory
  parser.add_argument("--train", required=False, help="trains a new model and saves it in the specified directory", action='store_true')
  parser.add_argument("--train-dir", required=False, help="trains a new model and saves it in the specified directory", default="model")
  parser.add_argument("--train-split", required=False, help="percent to split for train/test datasets. Defaults to 0.8.", default=0.8)

  # fetch Nest camera images and store them locally
  parser.add_argument("--nest", required=False, help="fetches last image from Nest cameras.", action='store_true')
  parser.add_argument("--repeat", required=False, help="keeps fetching new images with the delay specified in seconds.", default=-1)

  # Parse the command-line arguments.
  args = parser.parse_args()

  # Get the arguments.
  if args.nest:
    print("[RUN] Fetching Nest images")
    lib.nest.fetch(config, int(args.repeat))

  if args.train:
    print("[RUN] Training model at folder: '"+ str(args.train_dir) + "'")
    lib.train.train(model_dir=args.train_dir, train_split=float(args.train_split))
