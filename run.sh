#!/bin/bash

while getopts ":m:t:l:s:d:" opt; do
  case $opt in
  m)
    MODEL_NAME="$OPTARG" # a model name from huggingface hub
    ;;
  t)
    TYPE="$OPTARG" # one of 'standard', 'long', 'longformer', 'hierarchical'
    ;;
  l)
    LANG="$OPTARG" # one of 'de', 'fr', 'it'
    ;;
  s)
    SEED="$OPTARG" # integer: is also used for naming the run and the output_dir!
    ;;
  d)
    DEBUG="$OPTARG" # one of 'True' or 'False'
    ;;
  \?)
    echo "Invalid option -$OPTARG" >&5
    ;;
  esac
done

printf "Argument MODEL_NAME is %s\n" "$MODEL_NAME"
printf "Argument TYPE is %s\n" "$TYPE"
printf "Argument LANG is %s\n" "$LANG"
printf "Argument SEED is %s\n" "$SEED"
printf "Argument DEBUG is %s\n" "$DEBUG"

MAX_SAMPLES=100
# enable max samples in debug mode to make it run faster
[ "$DEBUG" == "True" ] && MAX_SAMPLES_ENABLED="--max_train_samples $MAX_SAMPLES --max_eval_samples $MAX_SAMPLES --max_predict_samples $MAX_SAMPLES"
[ "$DEBUG" == "True" ] && FP16="False" || FP16="True"      # disable fp16 in debug mode because it might run on cpu
[ "$DEBUG" == "True" ] && REPORT="none" || REPORT="all"    # disable wandb reporting in debug mode
[ "$DEBUG" == "True" ] && BASE_DIR="tmp" || BASE_DIR="sjp" # set other dir when debugging so we don't overwrite results

# IMPORTANT: For bigger models, very small total batch sizes did not work (4 to 8), for some even 32 was too small
TOTAL_BATCH_SIZE=64 # we made the best experiences with this (32 and below sometimes did not train well)
LR=3e-5             # Devlin et al. suggest somewhere in {1e-5, 2e-5, 3e-5, 4e-5, 5e-5}
NUM_EPOCHS=5

# Batch size for RTX 3090 for
# Distilbert: 64
# BERT-base: 16
# BERT-large: 8
# HierBERT/Longformer (input size 4096) Distilbert: 8?
# HierBERT/Longformer (input size 2048) BERT-base: 4
# HierBERT/Longformer (input size 1024) BERT-base: 8
# LongBERT (input size 2048) BERT-base: 2
# LongBERT (input size 1024) BERT-base: 4
# LongBERT (input size 2048) XLM-RoBERTa-base: 1
# LongBERT (input size 1024) XLM-RoBERTa-base: 2
if [[ "$TYPE" == "standard" ]]; then
  BATCH_SIZE=16
elif [[ "$TYPE" == "long" ]]; then
  if [[ "$MODEL_NAME" =~ roberta|camembert ]]; then
    BATCH_SIZE=1
  else
    BATCH_SIZE=2
  fi
else # either 'hierarchical' or 'longformer'
  BATCH_SIZE=4
fi

TEST_ON_SPECIAL_SPLITS="False"

MODE='train'                                            # Can be either 'train' or 'test'
[ "$MODE" == "train" ] && TRAIN="True" || TRAIN="False" # disable training if we are not in train mode

TRAIN_LANG=de

LABEL_IMBALANCE_METHOD=oversampling

# Compute variables based on settings above
MODEL=$MODEL_NAME-$TYPE
DIR=$BASE_DIR/$MODE/$MODEL/$LANG/$SEED
ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / BATCH_SIZE))                  # use this to achieve a sufficiently high total batch size
# Assign variables for enabling/disabling respective BERT version
[ "$TYPE" == "standard" ] && MAX_SEQ_LENGTH=512 || MAX_SEQ_LENGTH=2048 # how many tokens to consider as input (hierarchical/long: 2048 is enough for facts)

# Set this to a path to start from a saved checkpoint and to an empty string otherwise
CHECKPOINT=""
#CHECKPOINT=sjp/train/$MODEL/$TRAIN_LANG/$SEED
[ "$CHECKPOINT" == "" ] && MODEL_PATH="$MODEL_NAME" || MODEL_PATH=$CHECKPOINT

CMD="
python run_tc.py
  --problem_type single_label_classification
  --model_name_or_path $MODEL_PATH
  --run_name $MODE-$MODEL-$LANG-$SEED
  --output_dir $DIR
  --long_input_bert_type $TYPE
  --learning_rate $LR
  --seed $SEED
  --language $LANG
  --do_train $TRAIN
  --do_eval
  --do_predict
  --tune_hyperparams False
  --fp16 $FP16
  --fp16_full_eval $FP16
  --group_by_length
  --logging_strategy steps
  --evaluation_strategy epoch
  --save_strategy epoch
  --label_imbalance_method $LABEL_IMBALANCE_METHOD
  --gradient_accumulation_steps $ACCUMULATION_STEPS
  --eval_accumulation_steps $ACCUMULATION_STEPS
  --per_device_train_batch_size $BATCH_SIZE
  --per_device_eval_batch_size $BATCH_SIZE
  --max_seq_length $MAX_SEQ_LENGTH
  --num_train_epochs $NUM_EPOCHS
  --load_best_model_at_end
  --metric_for_best_model eval_loss
  --save_total_limit 10
  --report_to $REPORT
  --overwrite_output_dir True
  --overwrite_cache False
  --test_on_special_splits $TEST_ON_SPECIAL_SPLITS
  $MAX_SAMPLES_ENABLED
"
#  --label_smoothing_factor 0.1 \ # does not work with custom loss function
#  --resume_from_checkpoint $DIR/checkpoint-$CHECKPOINT
#  --metric_for_best_model eval_f1_macro # would be slightly better for imbalanced datasets

echo "Running command
$CMD
This output can be used to quickly run the command in the IDE for debugging"

# Actually execute the command
eval $CMD
