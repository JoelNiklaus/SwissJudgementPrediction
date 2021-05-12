MODEL_NAME="deepset/gbert-large"

python run_tc.py \
  --run_name $MODEL_NAME \
  --model_name_or_path $MODEL_NAME \
  --dataset_name joelito/ler \
  --output_dir ./sjp/$MODEL_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy steps \
  --fp16 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 5 \
  --logging_steps 1000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --save_total_limit 10

