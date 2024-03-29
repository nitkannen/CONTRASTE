# CONTRASTE: Supervised Contrastive Pre-training With Aspect-based Prompts For Aspect Sentiment Triplet Extraction
[[Paper: CONTRASTE: Supervised Contrastive Pre-training With Aspect-based Prompts For Aspect Sentiment Triplet Extraction ]](https://aclanthology.org/2023.findings-emnlp.807.pdf)

*Rajdeep Mukherjee, Nithish Kannen, Saurabh Kumar Pandey, Pawan Goyal* \
**Indian Institute of Technology Kharagpur** \
[Empirical Methods in Natural Language Processing (EMNLP 2023)](https://2023.emnlp.org/)


*TLDR: Contrastive Pre-Training to improve aspect-level sentiment understanding for ABSA*

## Abstract

Existing works on Aspect Sentiment Triplet
Extraction (ASTE) explicitly focus on developing more efficient fine-tuning techniques for
the task. Instead, our motivation is to come up
with a generic approach that can improve the
downstream performances of multiple ABSA
tasks simultaneously. Towards this, we present
CONTRASTE, a novel pre-training strategy
using CONTRastive learning to enhance the
ASTE performance. While we primarily focus on ASTE, we also demonstrate the advantage of our proposed technique on other ABSA
tasks such as ACOS, TASD, and AESC. Given
a sentence and its associated (aspect, opinion, sentiment) triplets, first, we design aspectbased prompts with corresponding sentiments
masked. We then (pre)train an encoder-decoder
model by applying contrastive learning on the
decoder-generated aspect-aware sentiment representations of the masked terms. For finetuning the model weights thus obtained, we
then propose a novel multi-task approach where
the base encoder-decoder model is combined
with two complementary modules, a taggingbased Opinion Term Detector, and a regressionbased Triplet Count Estimator. Exhaustive experiments on four benchmark datasets and a detailed ablation study establish the importance of
each of our proposed components as we achieve
new state-of-the-art ASTE results.

**[Note]** Code release is in progress. Stay tuned!!

![Alt text](https://github.com/nitkannen/CONTRASTE/blob/main/figures/CONTRASTE.png)


Environment setup:

```
conda install --file requirements.txt
```

To fine-tune for 15res ASTE task with the ASTE-MTL model (without pre-training), run the following:
 
 ```
sh scripts/finetune/14res_ASTE_MTL.sh
sh scripts/finetune/15res_ASTE_MTL.sh
sh scripts/finetune/16res_ASTE_MTL.sh
sh scripts/finetune/lap14_ASTE_MTL.sh
 ```

* change the 'task' and 'dataset' arguments in shell script to change the task.
* disable the 'use_regressor' and 'use_tagger' arguments from the shell script to run the ASTE-base model





To perform contrastive pre-training of the model and to save the checkpoints after certain epoch, run:

```
sh scripts/pretrain.sh
 ```



To fine-tune the contrastive pre-trained model for the ASTE task, run:

  ```
sh scripts/finetune/14res_CONTRASTE_MTL.sh
sh scripts/finetune/15res_CONTRASTE_MTL.sh
sh scripts/finetune/16res_CONTRASTE_MTL.sh
sh scripts/finetune/lap14_CONTRASTE_MTL.sh
 ```


 ## Effect of Contrastive Pre-training

 ![Alt text](https://github.com/nitkannen/CONTRASTE/blob/main/figures/viz_contrast_plot.png)

 ## Results
 ![Alt text](https://github.com/nitkannen/CONTRASTE/blob/main/figures/table_results.png)

 
  ## Packages Required
  
  * datasets
  * pytorch_lightning
  * sentencepiece
  * transformers

In order to set-up the the environment at once using conda, run the following:

```
conda install --file requirements.txt
```

# Citation

## If you find our work useful, please cite using:
```
@inproceedings{mukherjee-etal-2023-contraste,
    title = "{CONTRASTE}: Supervised Contrastive Pre-training With Aspect-based Prompts For Aspect Sentiment Triplet Extraction",
    author = "Mukherjee, Rajdeep  and
      Kannen, Nithish  and
      Pandey, Saurabh  and
      Goyal, Pawan",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.807",
    doi = "10.18653/v1/2023.findings-emnlp.807",
    pages = "12065--12080",
    abstract = "Existing works on Aspect Sentiment Triplet Extraction (ASTE) explicitly focus on developing more efficient fine-tuning techniques for the task. Instead, our motivation is to come up with a generic approach that can improve the downstream performances of multiple ABSA tasks simultaneously. Towards this, we present CONTRASTE, a novel pre-training strategy using CONTRastive learning to enhance the ASTE performance. While we primarily focus on ASTE, we also demonstrate the advantage of our proposed technique on other ABSA tasks such as ACOS, TASD, and AESC. Given a sentence and its associated (aspect, opinion, sentiment) triplets, first, we design aspect-based prompts with corresponding sentiments masked. We then (pre)train an encoder-decoder model by applying contrastive learning on the decoder-generated aspect-aware sentiment representations of the masked terms. For fine-tuning the model weights thus obtained, we then propose a novel multi-task approach where the base encoder-decoder model is combined with two complementary modules, a tagging-based Opinion Term Detector, and a regression-based Triplet Count Estimator. Exhaustive experiments on four benchmark datasets and a detailed ablation study establish the importance of each of our proposed components as we achieve new state-of-the-art ASTE results.",
}
```
