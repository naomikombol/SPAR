#!/bin/bash

set -e 

# Image generation config
IMAGES="cfg_SA1B_25k"
# Possible models: siglip2 maskclip dinov3txt
MODEL="maskclip" 

# Evaluation datasets
DATASETS=(
  "cfg_voc21"
  "cfg_voc20"
  "cfg_city_scapes"
  "cfg_ade20k"
  "cfg_context60"
  "cfg_context59"
)

RED="\033[31m"
GREEN="\033[32m"
RESET="\033[0m"


echo -e "${GREEN}====================================${RESET}"
echo -e "${RED}SPAR${RESET} ${GREEN}Pipeline${RESET}"
echo "Images: $IMAGES"
echo "Model: $MODEL"
echo -e "${GREEN}====================================${RESET}"


# Comment out steps as needed
# It is not necessary to always regenerate images

# 1. Generate images
echo ">>> Generating images $IMAGES"
python generate_images.py -config $IMAGES

# 2. Generate embeddings
echo ">>> Generating embeddings for $MODEL"
python generate_embeddings.py -model $MODEL 

# 3. Train SPAR-model
echo ">>> Training SPAR-$MODEL"
CHECKPOINT=$(python train.py -model $MODEL | tee /dev/stderr | grep "CHECKPOINT:" | cut -d':' -f2)
# Last trained checkpoint name gets captured for ease of evaluation
echo "Checkpoint file: $CHECKPOINT"

# 4. Evaluate SPAR-model over 6 datasets
echo ">>> Evaluation loop"
for DATASET in "${DATASETS[@]}"; do
    echo "------------------------------------"
    echo "Evaluating SPAR-$MODEL on $DATASET"
    echo "------------------------------------"

    python eval.py \
        -model $MODEL \
        -dataset $DATASET \
        -checkpoint $CHECKPOINT
done
