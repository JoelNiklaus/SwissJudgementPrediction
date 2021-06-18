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
# LongBERT (input size 2048) BERT-base: 1
# LongBERT (input size 1024) BERT-base: 2

# IMPORTANT: For bigger models, very small total batch sizes did not work (4 to 8), for some even 32 was too small
BASE_DIR=sjp
TYPE=long # one of standard, long, hierarchical
TOTAL_BATCH_SIZE=64
BATCH_SIZE=2        # depends on how much we can fit on the gpu
MAX_SEQ_LENGTH=2048 # how many tokens to consider as input (hierarchical, 2048 is enough for facts)
NUM_EPOCHS=5

# Compute variables based on settings above
MODEL=$MODEL_NAME-$TYPE
DIR=$BASE_DIR/$MODEL/$1
ACCUMULATION_STEPS=$(($TOTAL_BATCH_SIZE / $BATCH_SIZE)) # use this to achieve a sufficiently high total batch size
# Assign variables for enabling/disabling respective BERT version
[ "$TYPE" == "long" ] && LONG_BERT="True" || LONG_BERT="False"
[ "$TYPE" == "hierarchical" ] && HIER_BERT="True" || HIER_BERT="False"

python run_tc.py \
  --problem_type "single_label_classification" \
  --model_name_or_path $MODEL_NAME \
  --run_name $MODEL-$1 \
  --output_dir $DIR \
  --use_long_bert $LONG_BERT \
  --use_hierarchical_bert $HIER_BERT \
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
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --save_total_limit 10

#  --max_train_samples 100 \
#  --max_eval_samples 100 \
#  --max_predict_samples 100

#  --overwrite_output_dir \
#  --overwrite_cache \
#  --resume_from_checkpoint "$DIR/deepset/gbert-base/checkpoint-10000"
