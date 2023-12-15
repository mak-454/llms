#!/bin/bash
# shellcheck disable=SC2086

set -xe

# Step 0
pushd dreambooth || true

# Step 0 cont
# __preparation_start__
# TODO: If running on multiple nodes, change this path to a shared directory (ex: NFS)
export DATA_PREFIX="/home/default/mak/sdiff_artifacts"
export ORIG_MODEL_NAME="CompVis/stable-diffusion-v1-4"
export ORIG_MODEL_HASH="b95be7d6f134c3a9e62ee616f310733567f069ce"
export ORIG_MODEL_DIR="$DATA_PREFIX/model-orig"
export ORIG_MODEL_PATH="$ORIG_MODEL_DIR/models--${ORIG_MODEL_NAME/\//--}/snapshots/$ORIG_MODEL_HASH"
export TUNED_MODEL_DIR="$DATA_PREFIX/model-tuned"
export IMAGES_REG_DIR="$DATA_PREFIX/images-reg"
export IMAGES_OWN_DIR="$DATA_PREFIX/images-own"
export IMAGES_NEW_DIR="$DATA_PREFIX/images-new"
export IMAGES_RESULT_DIR="$DATA_PREFIX/images-result"
# TODO: Add more workers = number of gpus
export NUM_WORKERS=1

mkdir -p $ORIG_MODEL_DIR $TUNED_MODEL_DIR $IMAGES_REG_DIR $IMAGES_OWN_DIR $IMAGES_NEW_DIR $IMAGES_RESULT_DIR

cp -rf ./lego-cars/* "$IMAGES_OWN_DIR/"

# Unique token to identify our subject (e.g., a random lego vs. our unqtkn lego)
export UNIQUE_TOKEN="mylegocar"
export CLASS_NAME="legocar"


use_lora=true

# Step 1
python download_model.py --model_dir=$ORIG_MODEL_DIR --model_name=$ORIG_MODEL_NAME --revision=$ORIG_MODEL_HASH



# Generate more samples data
submissionid=`echo $RANDOM | md5sum | head -c 20; echo;`
echo "Generating more samples data - $submissionid"
d3x ray job submit --submission-id $submissionid --working-dir $PWD -- python generate.py \
--model_dir=$ORIG_MODEL_PATH \
--output_dir=$IMAGES_REG_DIR \
--prompts="photo of a $CLASS_NAME" \
--num_samples_per_prompt=200 \
--use_ray_data


submissionid=`echo $RANDOM | md5sum | head -c 20; echo;`
echo "Start LoRA finetuning job - $submissionid"

d3x ray job submit --submission-id $submissionid --working-dir $PWD -- python train.py \
  --use_lora \
  --model_dir=$ORIG_MODEL_PATH \
  --output_dir=$TUNED_MODEL_DIR \
  --instance_images_dir=$IMAGES_OWN_DIR \
  --instance_prompt="photo of $UNIQUE_TOKEN $CLASS_NAME" \
  --class_images_dir=$IMAGES_REG_DIR \
  --class_prompt="photo of a $CLASS_NAME" \
  --train_batch_size=2 \
  --lr=1e-4 \
  --num_epochs=4 \
  --max_train_steps=200 \
  --num_workers $NUM_WORKERS

# Clear new dir
rm -rf "$IMAGES_NEW_DIR"/*.jpg

submissionid=`echo $RANDOM | md5sum | head -c 20; echo;`
echo "Loading and testing the finetuned model - $submissionid"
d3x ray job submit --submission-id $submissionid --working-dir $PWD -- python generate.py \
--model_dir=$ORIG_MODEL_PATH \
--lora_weights_dir=$TUNED_MODEL_DIR \
--output_dir=$IMAGES_NEW_DIR \
--prompts="photo of a $UNIQUE_TOKEN $CLASS_NAME in a bucket" \
--num_samples_per_prompt=5

# Save artifact
mkdir -p $IMAGES_RESULT_DIR/artifacts
cp -f "$IMAGES_NEW_DIR"/0-*.jpg $IMAGES_RESULT_DIR/artifacts/example_out.jpg

# Exit
popd || true
