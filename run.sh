#MODEL_NAME="distilbert-base-german-cased"
#MODEL_NAME="deepset/gbert-base"
#MODEL_NAME="deepset/gbert-large"
#MODEL_NAME="deepset/gelectra-base"
#MODEL_NAME="deepset/gelectra-large"
#MODEL_NAME="bert-base-cased"
MODEL_NAME="bert-large-cased"
#MODEL_NAME="allenai/longformer-base-4096"
#MODEL_NAME="allenai/longformer-large-4096"
#MODEL_NAME="google/bigbird-roberta-base"
#MODEL_NAME="google/bigbird-roberta-large"

GRADIENT_ACCUMULATION_STEPS=8 # use this to achieve a sufficiently high total batch size
BATCH_SIZE=4                  # more does not fit on the gpu for the larger models
MAX_SEQ_LENGTH=512            # how many tokens to consider as input
LOGGING_STEPS=1000
NUM_EPOCHS=3
SEED=42

python run_tc.py \
  --model_name_or_path $MODEL_NAME \
  --run_name $MODEL_NAME \
  --output_dir ./sjp/$MODEL_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \
  --overwrite_cache \
  --fp16 \
  --evaluation_strategy "steps" \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --eval_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --logging_steps $LOGGING_STEPS \
  --save_steps $LOGGING_STEPS \
  --eval_steps $LOGGING_STEPS \
  --seed $SEED \
  --max_seq_length $MAX_SEQ_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --save_total_limit 10

#  --max_train_samples 1000 \
#  --max_eval_samples 1000 \
#  --max_predict_samples 1000
#  --resume_from_checkpoint "sjp/deepset/gbert-base/checkpoint-10000"
