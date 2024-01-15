# CONTRASTE: Supervised Contrastive Pre-training With Aspect-based Prompts For Aspect Sentiment Triplet Extraction
*Rajdeep Mukherjee, Nithish Kannen, Saurabh Kumar Pandey, Pawan Goyal* \
*Indian Institute of Technology Kharagpur* \
**Empirical Methods in Natural Language Processing (EMNLP 2023)**

*TLDR: Contrastive Pre-Training to improve aspect-level sentiment understanding for ABSA**

**[Note]** Code release is in progress. Stay tuned!!

![Alt text]([image link](https://github.com/nitkannen/CONTRASTE/blob/main/figures/CONTRASTE.png))



To pretrain the model and save the chekpoints of the pretrained models after certain epochs use:

```
sh scripts/pretrain.sh
     
 ```

To finetune for 15res ASTE task without pretraining use:
 
 ```
sh scripts/finetune/15res.sh
 
 ```
To finetune the pretrained model on the ASTE Task using a particular checkpoint use:
 
 ```
!python main.py --task 15res \
                --train_dataset_path 15res/train \
                --dev_dataset_path 15res/dev \
                --test_dataset_path 15res/test \
                --model_name_or_path models/contraste_model_after_2_epochs\
                --n_gpu 1 \
                --do_train \
                --do_eval \
                --train_batch_size 2 \
                --gradient_accumulation_steps 2 \
                --eval_batch_size 16 \
                --learning_rate 3e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --beta 0.2 \
                --alpha 0.8 \
                --model_weights models/contraste_model_after_2_epochs \
                --logger_name 15res_logs_regressor_tagger_contrast2.txt \
                --log_message regressor_and_tagger_2 \
     
 ```
 
 
  ### Packages Required
  
  * datasets
  * pytorch_lightning
  * sentencepiece
  * transformers
