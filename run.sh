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
  --train_languages=*)
    TRAIN_LANGUAGES="${1#*=}" # comma separated list of 'de', 'fr', 'it', 'en' (example: de,fr NO SPACE!)
    ;;
  --test_languages=*)
    TEST_LANGUAGES="${1#*=}" # one of 'de', 'fr', 'it', 'en'
    ;;
  --jurisdiction=*)
    JURISDICTION="${1#*=}" # one of 'switzerland', 'india', 'both'
    ;;
  --data_augmentation_type=*)
    DATA_AUGMENTATION_TYPE="${1#*=}" # one of 'no_augmentation', 'translation' or 'back_translation'
    ;;
  --train_sub_datasets=*)
    TRAIN_SUB_DATASETS="${1#*=}" # instances of LegalArea or OriginCanton such as civil_law or SG
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
printf "Argument TRAIN_LANGUAGES is \t\t %s\n" "$TRAIN_LANGUAGES"
printf "Argument TEST_LANGUAGES is \t\t %s\n" "$TEST_LANGUAGES"
printf "Argument TRAIN_SUB_DATASETS is \t\t %s\n" "$TRAIN_SUB_DATASETS"
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
TOTAL_BATCH_SIZE=64                 # we made the best experiences with this (32 and below sometimes did not train well)
NUM_EPOCHS=10                       # high enough to be save, we use EarlyStopping anyway, but sometimes it doesn't stop and the benefit of the epochs after 3-5 is very marginal
LABEL_IMBALANCE_METHOD=oversampling # this achieved the best results in our experiments
SEG_TYPE=block                      # one of sentence, paragraph, block, overlapping
OVERWRITE_CACHE=True                # IMPORTANT: Make sure to set this to true as soon as something with the data changes

# label smoothing cannot be used with a custom loss function
# 0.1/0.2 seemed to be the best in the setting adapters-xlm-roberta-base-hierarchical de,fr to it
[ "$LABEL_IMBALANCE_METHOD" == "class_weights" ] && LABEL_SMOOTHING_FACTOR=0.0 || LABEL_SMOOTHING_FACTOR=0.1

# Devlin et al. suggest somewhere in {1e-5, 2e-5, 3e-5, 4e-5, 5e-5},
[ "$TRAIN_TYPE" == "bitfit" ] && LR=5e-4   # 5e-4 higher learning rate for bitfit because there are less parameters
[ "$TRAIN_TYPE" == "adapters" ] && LR=5e-5 # 5e-5 somehow this is better for adapters. Just don't ask why!
[ "$TRAIN_TYPE" == "finetune" ] && LR=1e-5 # 1e-5 https://openreview.net/pdf?id=nzpLWnVAyah: RoBERTa apparently has a lot of instability with lr 3e-5

# Batch size for RTX 3090 for
# Distilbert: 32
# BERT-base: 16
# BERT-large: 8
# HierBERT (input size 2048) BERT-base: 4
# HierBERT (input size 1024) BERT-base: 8
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
else # either 'hierarchical' or 'efficient'
  BATCH_SIZE=4
fi
if [[ "$MODEL_NAME" =~ distilbert ]]; then
  BATCH_SIZE=$(($BATCH_SIZE * 2))
fi

# Compute variables based on settings above
MODEL=$MODEL_NAME-$MODEL_TYPE
RUN_DIR=$TRAIN_TYPE/$MODEL/$TRAIN_LANGUAGES/$SEED
OUTPUT_DIR=$BASE_DIR/$RUN_DIR
ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / BATCH_SIZE)) # use this to achieve a sufficiently high total batch size
# how many tokens to consider as input (hierarchical/long: 2048 is enough for facts)
[ "$MODEL_TYPE" == "standard" ] && MAX_SEQ_LEN=512 || MAX_SEQ_LEN=2048
[ "$SEG_TYPE" == "block" ] && MAX_SEG_LEN=512 || MAX_SEG_LEN=128
[ "$SEG_TYPE" == "sentence" ] && MAX_SEGMENTS=16 || MAX_SEGMENTS=4
# disable training if we are not in train mode
[ "$TRAIN_MODE" == "train" ] && TRAIN="True" || TRAIN="False"
# Set this to a path to start from a saved checkpoint and to an empty string otherwise
[ "$TRAIN_MODE" == "train" ] && MODEL_PATH="$MODEL_NAME" || MODEL_PATH="sjp/$RUN_DIR"

# Adapter Configs
# Italian: https://adapterhub.ml/adapters/ukp/xlm-roberta-base-it-wiki_pfeiffer/
# French: https://adapterhub.ml/adapters/ukp/bert-base-multilingual-cased-fr-wiki_pfeiffer/
# German: https://adapterhub.ml/adapters/ukp/bert-base-multilingual-cased-de-wiki_pfeiffer/, https://adapterhub.ml/adapters/ukp/xlm-roberta-base-de-wiki_pfeiffer/
# IMPORTANT: so far, there is no xlm-roberta-base adapter for French and no bert-base-multilingual-cased adapter for Italian
ADAPTER_CONFIG="houlsby"   # 'houlsby' or 'pfeiffer': 'houlsby' seemed to be slightly better in the setting adapters-xlm-roberta-base-hierarchical de,fr to it
ADAPTER_REDUCTION_FACTOR=4 # 2 and 4 seem to get the best results in the setting adapters-xlm-roberta-base-hierarchical de,fr to it

# For now disable lang adapters because it is too complicated and they are not available for all languages
#[ "$TEST_LANGUAGES" != "$TRAIN_LANGUAGES" ] && LOAD_LANG_ADAPTER="$LANGUAGE/wiki@ukp" || LOAD_LANG_ADAPTER="None"
#LOAD_LANG_ADAPTER="False" # Use this to disable loading of language adapters in all cases
#LANG_ADAPTER_CONFIG="pfeiffer"
#[ "$LANGUAGE" == "it" ] && LANG_ADAPTER_NON_LINEARITY="relu" || LANG_ADAPTER_NON_LINEARITY="gelu"
#LANG_ADAPTER_REDUCTION_FACTOR=2

CMD="python run_tc.py
  --problem_type single_label_classification
  --model_name $MODEL_NAME
  --model_name_or_path $MODEL_PATH
  --run_name $TRAIN_TYPE-$MODEL-$TRAIN_LANGUAGES-$TEST_LANGUAGES-$SEED
  --output_dir $OUTPUT_DIR
  --long_input_bert_type $MODEL_TYPE
  --learning_rate $LR
  --seed $SEED
  --train_languages $TRAIN_LANGUAGES
  --test_languages $TEST_LANGUAGES
  --do_train $TRAIN
  --do_eval
  --do_predict
  --tune_hyperparams False
  --fp16 $FP16
  --fp16_full_eval $FP16
  --group_by_length
  --pad_to_max_length
  --logging_strategy steps
  --evaluation_strategy epoch
  --save_strategy epoch
  --label_smoothing_factor $LABEL_SMOOTHING_FACTOR
  --label_imbalance_method $LABEL_IMBALANCE_METHOD
  --gradient_accumulation_steps $ACCUMULATION_STEPS
  --eval_accumulation_steps $ACCUMULATION_STEPS
  --per_device_train_batch_size $BATCH_SIZE
  --per_device_eval_batch_size $BATCH_SIZE
  --segmentation_type $SEG_TYPE
  --max_seq_len $MAX_SEQ_LEN
  --max_segments $MAX_SEGMENTS
  --max_seg_len $MAX_SEG_LEN
  --data_augmentation_type $DATA_AUGMENTATION_TYPE
  --jurisdiction $JURISDICTION
  --use_pretrained_model True
  --log_all_predictions True
  --num_train_epochs $NUM_EPOCHS
  --load_best_model_at_end
  --metric_for_best_model eval_f1_macro
  --early_stopping_patience 2
  --save_total_limit 5
  --report_to $REPORT
  --overwrite_output_dir
  --overwrite_cache $OVERWRITE_CACHE
  --train_sub_datasets $TRAIN_SUB_DATASETS
  --test_on_sub_datasets $SUB_DATASETS
  --train_type $TRAIN_TYPE
  --train_adapter
  --adapter_config $ADAPTER_CONFIG
  --adapter_reduction_factor $ADAPTER_REDUCTION_FACTOR
  $MAX_SAMPLES_ENABLED
"
#  --load_lang_adapter $LOAD_LANG_ADAPTER
#  --lang_adapter_config $LANG_ADAPTER_CONFIG
#  --lang_adapter_non_linearity $LANG_ADAPTER_NON_LINEARITY
#  --lang_adapter_reduction_factor $LANG_ADAPTER_REDUCTION_FACTOR
#  --language $LANGUAGE

echo "
Running the following command (this can be used to quickly run the command in the IDE for debugging):
$CMD"

# Actually execute the command
eval $CMD
