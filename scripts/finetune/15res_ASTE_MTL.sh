python finetuning/main.py \
       --task ASTE \
       --dataset 15res  \
       --train_dataset_path finetuning/data/15res/train \
       --dev_dataset_path finetuning/data/15res/dev \
       --test_dataset_path finetuning/data/15res/test \
       --model_name_or_path t5-base \
       --do_train \
       --do_eval  \
       --alpha 0.8 \
       --beta 0.4 \
       --train_batch_size 4  \
       --gradient_accumulation_steps 4  \
       --eval_batch_size 16  \
       --learning_rate 3e-4  \
       --num_train_epochs 20  \
       --regressor True  \
       --use_tagger True  \
       --logger_name 15res_base.txt  \
       --log_message 4_4_3e4_0.2 \
       --gpu_id 0