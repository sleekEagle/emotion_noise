from dataloaders.creamad import get_dataset, EMOTION_MAP
import torch
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor

from datasets import load_dataset, Audio
# minds = load_dataset("PolyAI/minds14", name="en-US", split="train")



accuracy = evaluate.load("accuracy")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def preprocess_function(examples):
    audio_arrays = [x for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

label2id, id2label = dict(), dict()
for i, label in enumerate(EMOTION_MAP.keys()):
    label2id[label] = str(i)
    id2label[str(i)] = label

#dataloader
data_path = r'C:\Users\lahir\code\CREMA-D\AudioWAV'
train_ds = get_dataset(data_path, split='train')
val_ds = get_dataset(data_path, split='val')

encoded_minds = train_ds.map(preprocess_function, batched=True)
pass


num_labels = len(EMOTION_MAP)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

training_args = TrainingArguments(
    output_dir=r'C:\Users\lahir\models\emo',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)



pass

# training_args = TrainingArguments(
#     output_dir="yelp_review_classifier",
#     eval_strategy="epoch",
#     push_to_hub=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     compute_metrics=compute_metrics,
# )
# trainer.train()



