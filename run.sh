#!/bin/bash

while [ $# -gt 0 ]; do
  case "$1" in
  --model_name=*)
    MODEL_NAME="${1#*=}" # a model name from huggingface hub
    ;;
  --type=*)
    TYPE="${1#*=}" # one of 'standard', 'long', 'hierarchical', 'longformer', 'bigbird'
    ;;
  --language=*)
    LANGUAGE="${1#*=}" # one of 'de', 'fr', 'it', 'all'
    ;;
  --train_language=*)
    TRAIN_LANGUAGE="${1#*=}" # one of 'de', 'fr', 'it', 'all'
    ;;
  --mode=*)
    MODE="${1#*=}" # either 'train' or 'test'
    ;;
  --sub_datasets=*)
    SUB_DATASETS="${1#*=}" # one of 'True' or 'False'
    ;;
  --seed=*)
    SEED="${1#*=}" # integer: is also used for naming the run and the output_dir!
    ;;
  --debug=*)
    DEBUG="${1#*=}" # one of 'True' or 'False'
    ;;
  *)
    printf "***************************\n"
    printf "* Error: Invalid argument.*\n"
    printf "***************************\n"
    exit 1
    ;;
  esac
  shift
done

printf "Argument MODEL_NAME is \t\t\t %s\n"      "$MODEL_NAME"
printf "Argument TYPE is \t\t\t %s\n"            "$TYPE"
printf "Argument LANGUAGE is \t\t\t %s\n"        "$LANGUAGE"
printf "Argument TRAIN_LANGUAGE is \t\t %s\n"    "$TRAIN_LANGUAGE"
printf "Argument MODE is \t\t\t %s\n"            "$MODE"
printf "Argument SUB_DATASETS is \t\t %s\n"      "$SUB_DATASETS"
printf "Argument SEED is \t\t\t %s\n"            "$SEED"
printf "Argument DEBUG is \t\t\t %s\n"           "$DEBUG"

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
LABEL_IMBALANCE_METHOD=oversampling

# Batch size for RTX 3090 for
# Distilbert: 32
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
else # either 'hierarchical', 'longformer' or 'bigbird'
  BATCH_SIZE=4
fi
if [[ "$MODEL_NAME" =~ distilbert ]]; then
  BATCH_SIZE=$(($BATCH_SIZE * 2))
fi

# Compute variables based on settings above
MODEL=$MODEL_NAME-$TYPE
DIR=$BASE_DIR/$MODE/$MODEL/$LANGUAGE
[ "$MODE" == "test" ] && DIR="$DIR/$TRAIN_LANGUAGE"
DIR=$DIR/$SEED
ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / BATCH_SIZE)) # use this to achieve a sufficiently high total batch size
# how many tokens to consider as input (hierarchical/long: 2048 is enough for facts)
[ "$TYPE" == "standard" ] && MAX_SEQ_LENGTH=512 || MAX_SEQ_LENGTH=2048
# disable training if we are not in train mode
[ "$MODE" == "train" ] && TRAIN="True" || TRAIN="False"
# Set this to a path to start from a saved checkpoint and to an empty string otherwise
[ "$MODE" == "train" ] && MODEL_PATH="$MODEL_NAME" || MODEL_PATH="sjp/train/$MODEL/$TRAIN_LANGUAGE/$SEED"

CMD="
python run_tc.py
  --problem_type single_label_classification
  --model_name_or_path $MODEL_PATH
  --run_name $MODE-$MODEL-$LANGUAGE-$TRAIN_LANGUAGE-$SEED
  --output_dir $DIR
  --long_input_bert_type $TYPE
  --learning_rate $LR
  --seed $SEED
  --evaluation_language $LANGUAGE
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
  --save_total_limit 3
  --report_to $REPORT
  --overwrite_output_dir True
  --overwrite_cache False
  --test_on_sub_datasets $SUB_DATASETS
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
