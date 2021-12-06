#!/bin/bash

while [ $# -gt 0 ]; do
  case "$1" in
  --train_type=*)
    TRAIN_TYPE="${1#*=}" # one of 'finetune', 'adapters' or 'bitfit'
    ;;
  --train_mode=*)
    TRAIN_MODE="${1#*=}" # either 'train' or 'test'
    ;;
  --model_name=*)
    MODEL_NAME="${1#*=}" # a model name from huggingface hub
    ;;
  --model_type=*)
    MODEL_TYPE="${1#*=}" # one of 'standard', 'long', 'hierarchical', 'efficient'
    ;;
  --train_language=*)
    TRAIN_LANGUAGE="${1#*=}" # one of 'de', 'fr', 'it', 'all'
    ;;
  --language=*)
    LANGUAGE="${1#*=}" # one of 'de', 'fr', 'it', 'all'
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

printf "Argument TRAIN_TYPE is \t\t\t %s\n" "$TRAIN_TYPE"
printf "Argument TRAIN_MODE is \t\t\t %s\n" "$TRAIN_MODE"
printf "Argument MODEL_NAME is \t\t\t %s\n" "$MODEL_NAME"
printf "Argument MODEL_TYPE is \t\t\t %s\n" "$MODEL_TYPE"
printf "Argument TRAIN_LANGUAGE is \t\t %s\n" "$TRAIN_LANGUAGE"
printf "Argument LANGUAGE is \t\t\t %s\n" "$LANGUAGE"
printf "Argument SUB_DATASETS is \t\t %s\n" "$SUB_DATASETS"
printf "Argument SEED is \t\t\t %s\n" "$SEED"
printf "Argument DEBUG is \t\t\t %s\n" "$DEBUG"

MAX_SAMPLES=100
# enable max samples in debug mode to make it run faster
[ "$DEBUG" == "True" ] && MAX_SAMPLES_ENABLED="--max_train_samples $MAX_SAMPLES --max_eval_samples $MAX_SAMPLES --max_predict_samples $MAX_SAMPLES"
[ "$DEBUG" == "True" ] && REPORT="none" || REPORT="all"    # disable wandb reporting in debug mode
[ "$DEBUG" == "True" ] && BASE_DIR="tmp" || BASE_DIR="sjp" # set other dir when debugging so we don't overwrite results
[ "$DEBUG" == "True" ] && FP16="False" || FP16="True"      # disable fp16 in debug mode because it might run on cpu

# IMPORTANT: For bigger models, very small total batch sizes did not work (4 to 8), for some even 32 was too small
TOTAL_BATCH_SIZE=64                                 # we made the best experiences with this (32 and below sometimes did not train well)
NUM_EPOCHS=10                                       # high enough to be save, we use EarlyStopping anyway
LABEL_IMBALANCE_METHOD=oversampling                 # this achieved the best results in our experiments
SEG_TYPE=block                                      # one of sentence, paragraph, block, overlapping
OVERWRITE_CACHE=True                                # IMPORTANT: Make sure to set this to true as soon as something with the data changes

# Devlin et al. suggest somewhere in {1e-5, 2e-5, 3e-5, 4e-5, 5e-5}, https://openreview.net/pdf?id=nzpLWnVAyah: RoBERTa apparently has a lot of instability with lr 3e-5
[ "$TRAIN_TYPE" == "bitfit" ] && LR=5e-4 || LR=1e-5 # lower lr for adapters and finetune

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
if [[ "$MODEL_TYPE" == "standard" ]]; then
  BATCH_SIZE=16
elif [[ "$MODEL_TYPE" == "long" ]]; then
  if [[ "$MODEL_NAME" =~ roberta|camembert ]]; then
    BATCH_SIZE=1
  else
    BATCH_SIZE=2
  fi
else # either 'hierarchical', 'efficient' or 'longformer'
  BATCH_SIZE=4
fi
if [[ "$MODEL_NAME" =~ distilbert ]]; then
  BATCH_SIZE=$(($BATCH_SIZE * 2))
fi

# Compute variables based on settings above
MODEL=$MODEL_NAME-$MODEL_TYPE
DIR=$BASE_DIR/$TRAIN_TYPE/$TRAIN_MODE/$MODEL/$TRAIN_LANGUAGE
DIR=$DIR/$SEED
ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / BATCH_SIZE)) # use this to achieve a sufficiently high total batch size
# how many tokens to consider as input (hierarchical/long: 2048 is enough for facts)
[ "$MODEL_TYPE" == "standard" ] && MAX_SEQ_LEN=512 || MAX_SEQ_LEN=2048
[ "$SEG_TYPE" == "block" ] && MAX_SEG_LEN=512 || MAX_SEG_LEN=128
[ "$SEG_TYPE" == "sentence" ] && MAX_SEGMENTS=16 || MAX_SEGMENTS=4
# disable training if we are not in train mode
[ "$TRAIN_MODE" == "train" ] && TRAIN="True" || TRAIN="False"
# Set this to a path to start from a saved checkpoint and to an empty string otherwise
[ "$TRAIN_MODE" == "train" ] && MODEL_PATH="$MODEL_NAME" || MODEL_PATH="sjp/$TRAIN_TYPE/train/$MODEL/$TRAIN_LANGUAGE/$SEED"

# Adapter Configs
# Italian: https://adapterhub.ml/adapters/ukp/xlm-roberta-base-it-wiki_pfeiffer/
# French: https://adapterhub.ml/adapters/ukp/bert-base-multilingual-cased-fr-wiki_pfeiffer/
# German: https://adapterhub.ml/adapters/ukp/bert-base-multilingual-cased-de-wiki_pfeiffer/, https://adapterhub.ml/adapters/ukp/xlm-roberta-base-de-wiki_pfeiffer/
# IMPORTANT: so far, there is no xlm-roberta-base adapter for French and no bert-base-multilingual-cased adapter for Italian
ADAPTER_CONFIG="houlsby"   # 'houlsby' or 'pfeiffer'
ADAPTER_REDUCTION_FACTOR=4 # default 16
[ "$LANGUAGE" != "$TRAIN_LANGUAGE" ] && LOAD_LANG_ADAPTER="$LANGUAGE/wiki@ukp" || LOAD_LANG_ADAPTER="None"
# LOAD_LANG_ADAPTER="None" # Use this to disable loading of language adapters in all cases
LANG_ADAPTER_CONFIG=pfeiffer
[ "$LANGUAGE" == "it" ] && LANG_ADAPTER_NON_LINEARITY="relu" || LANG_ADAPTER_NON_LINEARITY="gelu"
LANG_ADAPTER_REDUCTION_FACTOR=2

# TODO make adapter experiments without loading lang adapter

CMD="python run_tc.py
  --problem_type single_label_classification
  --model_name $MODEL_NAME
  --model_name_or_path $MODEL_PATH
  --run_name $TRAIN_TYPE-$TRAIN_MODE-$MODEL-$TRAIN_LANGUAGE-$LANGUAGE-$SEED
  --output_dir $DIR
  --long_input_bert_type $MODEL_TYPE
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
  --segmentation_type $SEG_TYPE
  --max_seq_len $MAX_SEQ_LEN
  --max_segments $MAX_SEGMENTS
  --max_seg_len $MAX_SEG_LEN
  --num_train_epochs $NUM_EPOCHS
  --load_best_model_at_end
  --metric_for_best_model eval_f1_macro
  --early_stopping_patience 2
  --save_total_limit 5
  --report_to $REPORT
  --overwrite_output_dir True
  --overwrite_cache $OVERWRITE_CACHE
  --test_on_sub_datasets $SUB_DATASETS
  --train_type $TRAIN_TYPE
  --train_adapter True
  --adapter_config $ADAPTER_CONFIG
  --adapter_reduction_factor $ADAPTER_REDUCTION_FACTOR
  --load_lang_adapter $LOAD_LANG_ADAPTER
  --lang_adapter_config $LANG_ADAPTER_CONFIG
  --lang_adapter_non_linearity $LANG_ADAPTER_NON_LINEARITY
  --lang_adapter_reduction_factor $LANG_ADAPTER_REDUCTION_FACTOR
  --language $LANGUAGE
  $MAX_SAMPLES_ENABLED
"
#  --label_smoothing_factor 0.1 \ # does not work with custom loss function
#  --resume_from_checkpoint $DIR/checkpoint-$CHECKPOINT
#  --metric_for_best_model eval_f1_macro # would be slightly better for imbalanced datasets

echo "
Running the following command (this can be used to quickly run the command in the IDE for debugging):
$CMD"

# Actually execute the command
eval $CMD
