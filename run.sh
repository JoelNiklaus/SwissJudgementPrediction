# IMPORTANT: $1 is the seed and also used for naming the run and the output_dir!

# German models
#MODEL_NAME="distilbert-base-german-cased"
MODEL_NAME="deepset/gbert-base"
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
#MODEL_NAME="xlm-roberta-base"
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

# IMPORTANT: For bigger models, very small total batch sizes did not work (4 to 8), for some even 32 was too small
BASE_DIR=sjp
DIR=$BASE_DIR/$MODEL_NAME/$1
TOTAL_BATCH_SIZE=64
BATCH_SIZE=2                                            # depends on how much we can fit on the gpu
ACCUMULATION_STEPS=$(($TOTAL_BATCH_SIZE / $BATCH_SIZE)) # use this to achieve a sufficiently high total batch size
MAX_SEQ_LENGTH=4096                                     # how many tokens to consider as input
NUM_EPOCHS=5

python run_tc_updated.py \
  --problem_type "single_label_classification" \
  --model_name_or_path $MODEL_NAME \
  --run_name $MODEL_NAME-$1 \
  --output_dir $DIR \
  --use_hierarchical_bert \
  --do_train \
  --do_eval \
  --do_predict \
  --fp16 \
  --logging_strategy "steps" \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --gradient_accumulation_steps $ACCUMULATION_STEPS \
  --eval_accumulation_steps $ACCUMULATION_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --seed $1 \
  --max_seq_length $MAX_SEQ_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --save_total_limit 10

#  --max_train_samples 100 \
#  --max_eval_samples 100 \
#  --max_predict_samples 100

#  --overwrite_output_dir \
#  --overwrite_cache \
#  --resume_from_checkpoint "$DIR/deepset/gbert-base/checkpoint-10000"
