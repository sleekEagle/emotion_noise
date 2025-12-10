from dataloaders.cremad_hf import get_hf_dataset, EMOTION_MAP, feature_extractor
import torch
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import AutoFeatureExtractor

from datasets import load_dataset, Audio


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

label2id, id2label = dict(), dict()
for i, label in enumerate(EMOTION_MAP.keys()):
    label2id[label] = str(i)
    id2label[str(i)] = label

#dataloader
train_ds, eval_ds = get_hf_dataset()

num_labels = len(EMOTION_MAP)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
)

# for param in model.wav2vec2.parameters():
#     param.requires_grad = False
# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

training_args = TrainingArguments(
    output_dir=r'C:\Users\lahir\models\emo',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-05,
    lr_scheduler_type ="linear",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    warmup_ratio=0.1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,

)

trainer.train()

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



