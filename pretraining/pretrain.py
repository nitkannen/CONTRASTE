import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import nn
import datetime
import json
import math
import os
import random
import time
import pprint
import string

from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
import datasets
import random
from transformers import AutoTokenizer
import tqdm
from tqdm import tqdm


import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import random_split
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import json
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import argparse
import sentencepiece
from collections import deque

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import nn

from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


class ABSADataset(Dataset):
    def __init__(self, data_samples, tokenizer):
        super(ABSADataset, self).__init__()
        self.tokenizer = tokenizer
        self.t5_input = [
            self.tokenizer(data_samples[i]["sentence"])
            for i in range(len(data_samples))
        ]
        with self.tokenizer.as_target_tokenizer():
            self.t5_decoder_input = [
                self.tokenizer(data_samples[i]["masked_target"])
                for i in range(len(data_samples))
            ]

        self.t5_input_ids = [
            self.t5_input[i]["input_ids"] for i in range(len(data_samples))
        ]
        self.t5_attention_mask = [
            self.t5_input[i]["attention_mask"] for i in range(len(data_samples))
        ]

        self.t5_decoder_input_labels = [
            self.t5_decoder_input[i]["input_ids"] for i in range(len(data_samples))
        ]
        self.t5_decoder_attention_mask = [
            self.t5_decoder_input[i]["attention_mask"] for i in range(len(data_samples))
        ]

        self.raw_texts = [data_samples[i]["sentence"] for i in range(len(data_samples))]

        self.labels = [data_samples[i]["label"] for i in range(len(data_samples))]
        self.len = len(data_samples)

    def __getitem__(self, index):
        return (
            self.t5_input_ids[index],
            self.t5_attention_mask[index],
            [0] + self.t5_decoder_input_labels[index],
            [0] + self.t5_decoder_attention_mask[index],
            self.labels[index],
            self.raw_texts[index],
        )

    def __len__(self):
        return self.len


def collate_fn(batch):
    (
        input_ids,
        attention_mask,
        decoder_input_labels,
        decoder_attention_mask,
        labels,
        raw_text,
    ) = zip(*batch)

    input_ids = pad_sequence(
        [torch.tensor(input) for input in (input_ids)], batch_first=True
    )
    attention_mask = pad_sequence(
        [torch.tensor(att) for att in (attention_mask)], batch_first=True
    )
    decoder_input_labels = pad_sequence(
        [torch.tensor(dec) for dec in (decoder_input_labels)], batch_first=True
    )
    decoder_attention_mask = pad_sequence(
        [torch.tensor(dec_att) for dec_att in (decoder_attention_mask)],
        batch_first=True,
    )
    labels = torch.tensor(labels)

    return (
        input_ids,
        attention_mask,
        decoder_input_labels,
        decoder_attention_mask,
        labels,
    )


def load_tokenizer(tokenizer_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    tokenizer.add_tokens(
        ["<aspect>", "<category>", "<opinion>", "<sentiment>"], special_tokens=True
    )

    return tokenizer


def load_model(model_checkpoint, tokenizer):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))

    return model.to(device)


def save_tokenizer(tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)


def save_model_checkpoint(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)


def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)

    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def get_optimizer_grouped_parameters(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def get_optimizer_scheduler(model, train_dataloader, epochs):
    total_steps = len(train_dataloader) * epochs
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)  # 2e-7
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


def has_opposite_labels(labels):
    return not (labels.sum().item() <= 1 or (1 - labels).sum().item() <= 1)


def _shift_right(input_ids):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)

    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()

    assert torch.all(
        shifted_input_ids >= 0
    ).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument(
        "--pretrain_datapath",
        default="15res",
        type=str,
        required=True,
        help="[15res, 14res, 16res, lap14]",
    )
    parser.add_argument(
        "--avg_window",
        type=int,
        default=10,
        help="The length of the running window to print loss.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="t5-base",
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="size_batch_for_contrastive"
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="contrastive pretrain epochs"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-7)

    args = parser.parse_args()

    if not os.path.exists("./models"):
        os.mkdir("./models")

    gpu_id = args.gpu_id

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    random_seed(args.seed)

    f = open(args.pretrain_datapath)

    contrast_examples = json.load(f)

    tokenizer = load_tokenizer(args.model_name_or_path)

    Contrastive_Dataset = ABSADataset(contrast_examples, tokenizer)

    loader = DataLoader(
        Contrastive_Dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
    )

    model = load_model(args.model_name_or_path, tokenizer)

    epochs = args.epochs
    lr = args.lr

    ########### PreTraining ######################################################

    current_step = 0
    contrast_criterion = SupConLoss()
    optimizer, scheduler = get_optimizer_scheduler(model, loader, epochs)
    window_stats = deque([], maxlen=args.avg_window)
    model.train()

    for epoch in range(1, epochs + 1):
        start = time.time()

        total_loss = 0

        pbar = tqdm(loader)

        pbar.set_description(
            f"Epoch: {epoch}, Running loss: {np.mean(list(window_stats)):.2f}"
        )

        for idx, batch in enumerate(pbar):
            (
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_att_ids,
                labels,
            ) = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            decoder_att_ids = decoder_att_ids.to(device)

            if has_opposite_labels(labels):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                )

                mask_position = torch.tensor(
                    np.where(decoder_input_ids.cpu().numpy() == 32099, 1, 0)
                ).to(device)

                masked_embeddings = outputs.decoder_hidden_states[
                    -1
                ] * mask_position.unsqueeze(2)
                sentence_embedding = torch.sum(masked_embeddings, axis=1)

                normalized_sentence_embeddings = sentence_embedding.to(device)

                similar_loss = contrast_criterion(
                    normalized_sentence_embeddings.unsqueeze(1), labels=labels
                )
                loss = similar_loss

                optimizer.zero_grad()
                loss.backward()
                total_loss += loss.item()

                optimizer.step()
                scheduler.step()

                current_step += 1

                window_stats.append(loss.item())

                if len(window_stats) == args.avg_window and idx % 10 == 0:
                    pbar.set_description(
                        f"Epoch: {epoch}, Running loss: {np.mean(list(window_stats)):.2f}"
                    )

            else:
                pass

        if (
            epoch == 1
            or epoch == 2
            or epoch == 6
            or epoch == 8
            or epoch == 10
            or epoch == 12
            or epoch == 14
        ):  ## needs editing
            name = "contraste_model_after_" + str(epoch) + "_epochs"
            save_path = os.path.join(os.getcwd(), "models")
            model_path = os.path.join(save_path, name)
            save_model_checkpoint(model, model_path)
            save_tokenizer(tokenizer, model_path)

        print(
            "#########################################################################"
        )
        end = time.time()
        print("[Epoch {:2d}] complete in {:.2f} seconds".format(epoch, end - start))
        print(f"Loss at epoch: {total_loss / ( len(loader) *  args.batch_size) }")

        ####### Save models in mode directory
