# IMPORTANT: $1 is the seed and also used for naming the run and the output_dir!

# German models
#MODEL_NAME="distilbert-base-german-cased"
#MODEL_NAME="deepset/gbert-base"
#MODEL_NAME="deepset/gbert-large"
#MODEL_NAME="deepset/gelectra-base"
#MODEL_NAME="deepset/gelectra-large"

# French models
#MODEL_NAME="flaubert/flaubert_base_cased"
#MODEL_NAME="flaubert/flaubert_large_cased"
#MODEL_NAME="camembert/camembert-base-ccnet"
#MODEL_NAME="camembert/camembert-large"

# Italian models
#MODEL_NAME="dbmdz/bert-base-italian-cased"

# Multilingual models
#MODEL_NAME="distilbert-base-multilingual-cased"
#MODEL_NAME="bert-base-multilingual-cased"
MODEL_NAME="xlm-roberta-base"
#MODEL_NAME="xlm-roberta-large"

# English models
#MODEL_NAME="bert-base-cased"
#MODEL_NAME="bert-large-cased"
#MODEL_NAME="allenai/longformer-base-4096"      #needs debugging
#MODEL_NAME="allenai/longformer-large-4096"
#MODEL_NAME="google/bigbird-roberta-base"
#MODEL_NAME="google/bigbird-roberta-large"

# Batch size for RTX 3090 for
# Distilbert: 64
# BERT-base: 16
# BERT-large: 4?, 8?
# HierBERT (input size 4096) Distilbert: 4?
# HierBERT (input size 4096) BERT-base: 2
# LongBERT (input size 2048) BERT-base: 2
# LongBERT (input size 1024) BERT-base: 4

DEBUG=False
MAX_SAMPLES=100
# enable max samples in debug mode to make it run faster
[ "$DEBUG" == "True" ] && MAX_SAMPLES_ENABLED="--max_train_samples $MAX_SAMPLES --max_eval_samples $MAX_SAMPLES --max_predict_samples $MAX_SAMPLES"
[ "$DEBUG" == "True" ] && FP16="False" || FP16="True"   # disable fp16 in debug mode because it might run on cpu
[ "$DEBUG" == "True" ] && REPORT="none" || REPORT="all" # disable wandb reporting in debug mode

# IMPORTANT: For bigger models, very small total batch sizes did not work (4 to 8), for some even 32 was too small
BASE_DIR='sjp'
TYPE='standard'     # one of 'standard', 'long', 'hierarchical'
MAX_SEQ_LENGTH=512 # how many tokens to consider as input (hierarchical/long: 2048 is enough for facts)
LR=3e-5             # Devlin et al. suggest somewhere in {1e-5, 2e-5, 3e-5, 4e-5, 5e-5}
BATCH_SIZE=2        # depends on how much we can fit on the gpu
TOTAL_BATCH_SIZE=64
NUM_EPOCHS=5
SEED=$1
LANG='fr'

# Compute variables based on settings above
MODEL=$MODEL_NAME-$TYPE
DIR=$BASE_DIR/$MODEL/$SEED
ACCUMULATION_STEPS=$(($TOTAL_BATCH_SIZE / $BATCH_SIZE)) # use this to achieve a sufficiently high total batch size
# Assign variables for enabling/disabling respective BERT version
[ "$TYPE" == "long" ] && LONG_BERT="True" || LONG_BERT="False"
[ "$TYPE" == "hierarchical" ] && HIER_BERT="True" || HIER_BERT="False"

MODE='evaluate' # Can be either 'train' or 'evaluate
CHECKPOINT=3140
[ "$MODE" == "evaluate" ] && RESUME_FROM_CHECKPOINT="--resume_from_checkpoint $DIR/checkpoint-$CHECKPOINT"
[ "$MODE" == "train" ] && TRAIN="True" || TRAIN="FALSE" # disable training if we are not in train mode

python run_tc.py \
  --problem_type "single_label_classification" \
  --model_name_or_path $MODEL_NAME \
  --run_name $MODEL-$SEED \
  --output_dir $DIR \
  --use_long_bert $LONG_BERT \
  --use_hierarchical_bert $HIER_BERT \
  --learning_rate $LR \
  --seed $SEED \
  --language $LANG \
  --do_train $TRAIN \
  --do_eval \
  --do_predict \
  --tune_hyperparams False \
  --fp16 $FP16 \
  --fp16_full_eval $FP16 \
  --group_by_length \
  --logging_strategy "steps" \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --eval_accumulation_steps $ACCUMULATION_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --save_total_limit 10 \
  --report_to $REPORT \
  --overwrite_output_dir False \
  --overwrite_cache False \
  $MAX_SAMPLES_ENABLED \
  $RESUME_FROM_CHECKPOINT

#  --label_smoothing_factor 0.1 \ # does not work with custom loss function
