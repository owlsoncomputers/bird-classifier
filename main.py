#imports
import datasets
import random
import matplotlib.pyplot as plt
import evaluate
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import numpy as np
import torch
from PIL import Image




# load dataset
dataset = datasets.load_dataset("bentrevett/caltech-ucsd-birds-200-2011")


#decoding labels
labels = dataset["train"].features["label"].names
labelsToID, IDToLabel = dict(), dict()

for i, label in enumerate(labels):
    labelsToID[label] = i
    IDToLabel[i] = label

#show example image
example = dataset["train"][0]
image = example["image"]
plt.title(IDToLabel[example["label"]])
plt.imshow(image)
plt.show()


#load accuracy model
metric = evaluate.load("accuracy")


#load model
model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"
batch_size = 32


#preprocessing
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
print("Image Processor:")
print(image_processor)

normalize = Normalize(mean= image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")


train_transforms = Compose([RandomResizedCrop(crop_size), RandomHorizontalFlip(), ToTensor(), normalize,])
val_transforms = Compose([Resize(size), CenterCrop(crop_size), ToTensor(), normalize,])

def preprocessing_train(batch):
    batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in batch["image"]]
    return batch

def preprocessing_val(batch):
    batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in batch["image"]]
    return batch

splits = dataset["train"].train_test_split(test_size=0.1)
train_data = splits["train"]
val_data = splits['test']

train_data.set_transform(preprocessing_train)
val_data.set_transform(preprocessing_val)

print(train_data[0])

#training
model = AutoModelForImageClassification.from_pretrained(model_checkpoint, label2id=labelsToID, id2label=IDToLabel, ignore_mismatched_sizes = True)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-eurosat",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

#training

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

image_processor = AutoImageProcessor.from_pretrained("swin-tiny-patch4-window7-224-finetuned-eurosat")
model = AutoModelForImageClassification.from_pretrained("swin-tiny-patch4-window7-224-finetuned-eurosat")


testImage3 = Image.open("test1.png")

encoding = image_processor(testImage3.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits




predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
plt.title(model.config.id2label[predicted_class_idx])
plt.imshow(testImage3)
plt.show()