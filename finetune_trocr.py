"""
Fine-tuning do TrOCR com seu próprio dataset de escrita à mão.

COMO USAR
1. Descompacte o .zip (gerado pela ferramenta de coleta, ou exportado a partir
   do Krita) numa pasta. Ela deve conter as imagens (ex: linha_001.png) mais
   um arquivo metadata.csv com as colunas "file_name" e "text".
2. Ajuste as variáveis da seção CONFIGURAÇÃO abaixo.
3. Instale as dependências:
   pip install transformers datasets evaluate jiwer pillow torch --break-system-packages
4. Rode:
   python finetune_trocr.py
"""

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    ViTImageProcessor,
    RobertaTokenizer,
)
import evaluate

# ========================= CONFIGURAÇÃO =========================
DATASET_DIR = "./"                # pasta com metadata.csv + imagens
OUTPUT_DIR = "./trocr-minha-letra"       # onde o modelo treinado será salvo

# Checkpoint de partida. O padrão é treinado em manuscrito (inglês), o que dá
# uma boa base visual para lidar com caligrafia. Como seu vocabulário de
# treino é pequeno e fechado (as frases que você escreveu), o decodificador
# consegue se adaptar ao português mesmo partindo de uma base em inglês.
# Se os resultados ficarem estranhos no texto gerado, vale testar
# "mazafard/trocr-finetuned-portugues" (base treinada em texto impresso em
# português — bom vocabulário, mas encoder visual não é voltado a manuscrito).
BASE_MODEL = "microsoft/trocr-base-handwritten"

MAX_TARGET_LENGTH = 64      # tokens; suas frases são curtas, isso sobra
EVAL_FRACTION = 0.1         # % dos dados reservado para validação
EPOCHS = 30                 # dataset pequeno -> mais épocas ajuda; acompanhe o CER
BATCH_SIZE = 4              # reduza se faltar memória de GPU
LEARNING_RATE = 5e-5
# ==================================================================


def carregar_splits():
    df = pd.read_csv(os.path.join(DATASET_DIR, "metadata.csv"))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # embaralha
    corte = int(len(df) * (1 - EVAL_FRACTION))
    return df.iloc[:corte].reset_index(drop=True), df.iloc[corte:].reset_index(drop=True)


class HandwritingDataset(Dataset):
    def __init__(self, df, processor, root_dir, max_target_length):
        self.df = df
        self.processor = processor
        self.root_dir = root_dir
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        text = str(self.df["text"][idx])

        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length, truncation=True
        ).input_ids
        # tokens de padding não devem contar na loss
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def main():
    image_processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # configuração necessária pro modelo gerar texto corretamente
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # parâmetros de geração (transformers 5.x exige generation_config)
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = MAX_TARGET_LENGTH
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    train_df, eval_df = carregar_splits()
    print(f"Treino: {len(train_df)} frases | Validação: {len(eval_df)} frases")

    train_dataset = HandwritingDataset(train_df, processor, DATASET_DIR, MAX_TARGET_LENGTH)
    eval_dataset = HandwritingDataset(eval_df, processor, DATASET_DIR, MAX_TARGET_LENGTH)

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nModelo salvo em: {OUTPUT_DIR}")

    # teste rápido com uma frase da validação
    if len(eval_df) > 0:
        exemplo = eval_df.iloc[0]
        imagem = Image.open(os.path.join(DATASET_DIR, exemplo["file_name"])).convert("RGB")
        pixel_values = processor(imagem, return_tensors="pt").pixel_values.to(model.device)
        generated_ids = model.generate(pixel_values)
        texto_previsto = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("\n--- Teste rápido ---")
        print(f"Esperado : {exemplo['text']}")
        print(f"Previsto : {texto_previsto}")


if __name__ == "__main__":
    main()
