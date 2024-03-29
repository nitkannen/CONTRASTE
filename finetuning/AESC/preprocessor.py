from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import os
import torch

sent_map = {}
sent_map["POS"] = "positive"
sent_map["NEU"] = "neutral"
sent_map["NEG"] = "negative"


def read_data(path, k_shot):
    sents = open(path + ".sent", "r")
    sentences = sents.readlines()
    tups = open(path + ".tup", "r")
    tuples = tups.readlines()

    return sentences, tuples


def generate_target(d):
    """
    takes a aspect triple dictionary and linearizes it
    """
    summary = ""
    if len(d) == 0:
        return summary
    triplets = []
    for items in d:
        triplets.append(f"<aspect> {items[0]} <sentiment> {items[1]}")

    summary = " [SSEP] ".join(triplets)

    # print("Generate target : ", summary.strip())
    return summary.strip()


def generate_triplet_dict(tuples, sentence):
    """
    takes a set of tuples and generates triplet dictionary
    """
    triplets = tuples.split("|")
    d = []
    ordered_triplets = []
    for triplet in triplets:
        # print(triplet)
        a, s = triplet.split(";")
        ordered_triplets.append(
            (sentence.find(a.strip()), sentence.find(s.strip()), triplet)
        )
    # print(ordered_triplets)
    ordered_triplets = sorted(ordered_triplets)
    # print(ordered_triplets)

    for triplet in ordered_triplets:
        a, s = triplet[2].split(";")
        d.append((a.strip(), s.strip()))

    # print("Generate triplet dict : ", d)
    return d


def get_transformed_data(sentences_list, tuples_list):
    """
    Preprocess the raw data into Generative Targets
    """
    inputs = []
    targets = []

    for i in range(len(sentences_list)):
        sent = sentences_list[i].strip()
        tup = tuples_list[i]
        tup_dict = generate_triplet_dict(tup, sent)
        target = generate_target(tup_dict)
        inputs.append(sent)
        targets.append(target)
    # print("Targets : ", targets)

    return inputs, targets


class ABSA_Dataset(Dataset):
    def __init__(self, tokenizer, data_path, task, k_shot=-1, max_len=128):
        # 'data/aste/rest16/train.txt'
        self.data_path = data_path
        self.task = task
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.k_shot = k_shot

        self.inputs = []
        self.targets = []
        self.input_tags = []
        self.trip_counts = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze
        target_mask = self.targets[index][
            "attention_mask"
        ].squeeze()  # might need to squeeze
        # op_tags = self.input_tags[index].squeeze()
        triplet_count = self.trip_counts[index]

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "triplet_count": triplet_count,
        }

    def get_tags(self, text, tuples):
        # tuples.split('|')
        triplets = tuples.split("|")
        target_tokens = []
        for triplet in triplets:
            a, o = triplet.split(";")
            target_tokens.append(o.strip())

        tokens = self.tokenizer.tokenize(text.strip())
        target = [0 for i in range(len(tokens))]

        for target_token in target_tokens:
            print("Target token : ", target_token)
            sub_tok = self.tokenizer.tokenize(target_token)

            if len(sub_tok) == 0:
                continue

            for idx in range(len(tokens) + 1 - len(sub_tok)):
                start_token = tokens[idx]
                match = True

                if sub_tok[0] == start_token:
                    for j in range(idx, idx + len(sub_tok)):
                        if sub_tok[j - idx] != tokens[j]:
                            match = False
                    if match:
                        print("BBBBBB")
                        target[idx] = 1  ########## 1 == 'B'
                        for k in range(idx + 1, idx + len(sub_tok)):
                            print("IIIIII")
                            target[k] = 2  ###########  2 == 'I'

        print("Tags : ", target)
        return target

    def get_all_tags(self, sentences_list, tuples_list):
        """
        Preprocess the raw data into tags for opinion
        """
        tags = []

        for i in range(len(sentences_list)):
            sent = sentences_list[i].strip()
            tup = tuples_list[i]
            t = self.get_tags(sent, tup)
            tags.append(t)

        print("All Tags : ", tags)
        return (tags,)

    def count_triplets(self, tuples_list):
        trip_count = []

        for i in range(len(tuples_list)):
            trip_count.append(len(tuples_list[i].split("|")))

        return trip_count

    def _build_examples(self):
        sentences, tuples = read_data(self.data_path, self.k_shot)
        inputs, targets = get_transformed_data(sentences, tuples)
        # input_tags = self.get_all_tags(sentences, tuples)
        trip_counts = self.count_triplets(tuples)

        for i in range(len(inputs)):
            input = inputs[i]
            target = targets[i]
            # input_tag = input_tags[i]
            trip_count = trip_counts[i]

            tokenized_input = self.tokenizer(
                [input],
                max_length=self.max_len,
                pad_to_max_length=True,
                truncation=True,
                return_tensors="pt",
            )

            with self.tokenizer.as_target_tokenizer():
                tokenized_target = self.tokenizer(
                    [target],
                    max_length=self.max_len,
                    pad_to_max_length=True,
                    truncation=True,
                    return_tensors="pt",
                )

            # input_tag = input_tag + [0] * (self.max_len - len(input_tag))
            # print(len(input_tag), len(input_tag[0]))
            # input_tag = torch.tensor(input_tag)
            trip_count = torch.tensor(trip_count)

            # self.input_tags.append(input_tag)
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
            self.trip_counts.append(trip_count)
