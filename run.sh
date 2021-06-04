# IMPORTANT: $1 is the seed and also used for naming the run and the output_dir!

# German models
MODEL_NAME="distilbert-base-german-cased"  # bs 32
#MODEL_NAME="deepset/gbert-base"            # bs 32
#MODEL_NAME="deepset/gbert-large"           # bs 32
#MODEL_NAME="deepset/gelectra-base"         # bs 32
#MODEL_NAME="deepset/gelectra-large"        # bs 64

# Multilingual models
#MODEL_NAME="distilbert-base-multilingual-cased"  # bs 32
#MODEL_NAME="bert-base-multilingual-cased"        # bs 32

# English models
#MODEL_NAME="bert-base-cased"                   # bs 32
#MODEL_NAME="bert-large-cased"                  # bs 64
#MODEL_NAME="allenai/longformer-base-4096"      # bs 64, needs debugging
#MODEL_NAME="allenai/longformer-large-4096"     # bs
#MODEL_NAME="google/bigbird-roberta-base"        # bs 64
#MODEL_NAME="google/bigbird-roberta-large"      # bs

# IMPORTANT: For bigger models, very small total batch sizes did not work (4 to 8), for some even 32 was too small
BASE_DIR=sjp
DIR=$BASE_DIR/$MODEL_NAME/$1
ACCUMULATION_STEPS=4         # use this to achieve a sufficiently high total batch size
BATCH_SIZE=16                 # more does not fit on the gpu for the larger models
MAX_SEQ_LENGTH=512           # how many tokens to consider as input
NUM_EPOCHS=5

python run_tc.py \
  --model_name_or_path $MODEL_NAME \
  --run_name $MODEL_NAME-$1 \
  --output_dir $DIR \
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

#  --max_train_samples 1000 \
#  --max_eval_samples 1000 \
#  --max_predict_samples 1000
#  --overwrite_output_dir \
#  --overwrite_cache \

#  --resume_from_checkpoint "$DIR/deepset/gbert-base/checkpoint-10000"
