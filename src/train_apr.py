import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer, TrainerCallback
from apr import FullAPRModel
from data_collector_ctc_with_padding import data_collator
from prepare_dataset import get_or_create_processed_dataset, processor, vocab_dict


class UnfreezeCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        target_model = model.module if hasattr(model, 'module') else model
        
        if state.epoch < 1:
            for name, param in target_model.wav2vec2.named_parameters():
                param.requires_grad = False
            print("Starting from scratch. Wav2Vec encoder is frozen.")
        else:
            for name, param in target_model.wav2vec2.named_parameters():
                if "feature_extractor" not in name:
                    param.requires_grad = True
            print(f"Resumed at epoch {state.epoch}. Wav2Vec encoder is unfrozen.")

    def on_epoch_end(self, args, state, control, model, **kwargs):
        if round(state.epoch) == 1:
            target_model = model.module if hasattr(model, 'module') else model
            
            for name, param in target_model.wav2vec2.named_parameters():
                if "feature_extractor" not in name:
                    param.requires_grad = True
                    
            print("Epoch 1 ended. Unfreezing the Wav2Vec encoder.")

def main():
    prepared_dataset = get_or_create_processed_dataset()
    split_dataset = prepared_dataset.train_test_split(test_size=0.1)
    
    model = FullAPRModel(
        model_name="NbAiLab/nb-wav2vec2-300m-bokmaal-v2",
        vocab_size=len(vocab_dict),
        pad_token_id=vocab_dict["<pad>"],
        start_token_id=vocab_dict["<start>"],
        end_token_id=vocab_dict["<end>"]
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = vocab_dict["<pad>"]
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    training_args = TrainingArguments(
        output_dir="./checkpoints/e2er-apr-stage",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,     
        eval_strategy="steps",
        num_train_epochs=20,           
        fp16=True,                     
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir="./logs_apr"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=processor,
        callbacks=[UnfreezeCallback()]  # << Attach the unfreeze logic here
    )

    trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    main()