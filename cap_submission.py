import json
import os
from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from transformers import default_data_collator
# ##########
# TODO: Add more imports
import pandas as pd
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    GPT2Tokenizer,
    default_data_collator
)
# ##########

class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = "google/vit-base-patch16-224-in21k"
    decoder = "gpt2"

    # Dataset path
    root_dir = "./flickr8k"

    name = f"cap-vlm-model"

    # Hyperparameters
    batch_size = 16
    lr = 5e-5
    epochs = 5

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 45     # TODO: Can play around


    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 50

    eval_steps = 500
    save_steps = 1000
    warmup_steps = 1000

class FlickrDataset(Dataset):
    def __init__(
        self,
        args,
        processor,
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val", "test"]
        self.args = args
        # ####################
        # TODO: Load Flickr8k dataset
        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer
        self.processor = processor
        self.tokenizer = tokenizer

        df = pd.read_csv(
            os.path.join(args.root_dir, f"{mode}.txt"),
            sep=';',
            encoding='utf-8'
        )

        # Skip header if present
        if df.columns[0] == 'image':
            df = df.iloc[1:]

        # Store image paths and captions
        self.img_paths = df['image'].values
        self.captions = df['caption'].values
        # ####################

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ####################
        image_path = os.path.join(self.args.root_dir, "images", self.img_paths[idx])
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Process caption
        caption = self.captions[idx]
        # Add special tokens for GPT-2 format
        caption = f"<|beginoftext|>{caption}<|endoftext|>"

        labels = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.args.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        encoding = {
            "pixel_values": pixel_values,       # Return processed image as a tensor
            "labels": labels,             # Return tokenized caption as a padded tensor
            "path": self.img_paths[idx],
            "captions": self.captions[idx],
        }
        # ####################

        return encoding


def train_cap_model(args):
    # Define your vision processor and language tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder)
    processor = ViTImageProcessor.from_pretrained(args.encoder)


    # Define your Image Captioning model using Vision-Encoder-Decoder model
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'bos_token': '<|beginoftext|>', 'pad_token': '<|pad|>'}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder,
        args.decoder
    )   # NOTE: Send your model to GPU

    model.decoder.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model = model.cuda()

    # Modify the embedding lookup table in decoder model and the tokenizer
    # to include bos_token "<|beginoftext|>" and pad_token "<|pad|>"
    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"



    # Load train/val dataset
    train_dataset = FlickrDataset(args, processor, tokenizer, "train")
    val_dataset = FlickrDataset(args, processor, tokenizer,"val")

    # Model configuration.
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.generation_config.max_length = args.max_length #None
    model.generation_config.num_beams = args.num_beams #None

    # TODO: Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        predict_with_generate=True,
        save_total_limit=2,
        # Add gradient clipping to prevent explosion
        max_grad_norm=1.0,
    )

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()
    trainer.save_model(args.name)


def load_trained_model(
    ckpt_dir: str,
    ):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    processor = ViTImageProcessor.from_pretrained(
        config["encoder_name"] if "encoder_name" in config
        else "google/vit-base-patch16-224-in21k"
    )

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_dir)

    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer

def eval_model(model_path):
    model, processor, tokenizer = load_trained_model(model_path)

    args = Args()
    val_dataset = FlickrDataset(args, processor, tokenizer, "val")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.name,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True
    )

    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator
    )

    metrics = trainer.evaluate()
    print(metrics)

def inference(
    img_path,
    model,
    processor,
    tokenizer,
    ):

    image = Image.open(img_path).convert("RGB")
    img_tensor = processor(image, return_tensors="pt").pixel_values

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    #  Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = model.generate(
        img_tensor,
        max_length=model.generation_config.max_length,
        num_beams=model.generation_config.num_beams,
        early_stopping=True
    )

    # Tokens -> Str
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """NOTE: DO NOT CHANGE.
    Compute BLEU score.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }
