import numpy as np
import evaluate
from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer
from data_collector_ctc_with_padding import data_collator
from prepare_dataset import get_or_create_processed_dataset, processor, vocab_dict

def main():
    prepared_dataset = get_or_create_processed_dataset()

    split_dataset = prepared_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    model = Wav2Vec2ForCTC.from_pretrained(
        "NbAiLab/nb-wav2vec2-300m-bokmaal-v2",
        ignore_mismatched_sizes=True,
        vocab_size=len(vocab_dict),
        pad_token_id=processor.tokenizer.pad_token_id
    )

    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = TrainingArguments(
        output_dir="./checkpoints/wav2vec2-300m-phoneme-ctc",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=5,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_first_step=True,
        report_to="tensorboard",
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor, 
    )

    trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    main()