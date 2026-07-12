"""
Inferência com o modelo TrOCR puro (pré-treinado), sem fine-tuning.

Útil para comparar com o modelo fine-tunado e avaliar o ganho do treinamento.

USO:
    # Uma imagem:
    python inferir_base.py caminho/para/imagem.jpg

    # Várias imagens:
    python inferir_base.py imagem1.jpg imagem2.png imagem3.JPEG

    # Todas as imagens de uma pasta:
    python inferir_base.py frases/*.JPEG
"""

import sys
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor
from transformers import RobertaTokenizer

# Modelo pré-treinado do Hugging Face (handwritten)
PRETRAINED_MODEL = "microsoft/trocr-base-handwritten"
# O decoder do TrOCR-base-handwritten usa o vocabulário do roberta-large
DECODER_TOKENIZER = "FacebookAI/roberta-large"


def carregar_modelo(model_name):
    """Carrega o modelo e o processor pré-treinados do Hugging Face."""
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(DECODER_TOKENIZER)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return processor, model, device


def inferir(image_path, processor, model, device):
    """Recebe o caminho de uma imagem e retorna o texto previsto."""
    imagem = Image.open(image_path).convert("RGB")
    pixel_values = processor(imagem, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=200)

    texto = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return texto


def main():
    if len(sys.argv) < 2:
        print("Uso: python inferir_base.py <imagem1> [imagem2] [imagem3] ...")
        print("Exemplo: python inferir_base.py frases/image1.JPEG")
        sys.exit(1)

    caminhos = sys.argv[1:]

    print(f"Carregando modelo pré-treinado: {PRETRAINED_MODEL}")
    processor, model, device = carregar_modelo(PRETRAINED_MODEL)
    print(f"Modelo carregado no dispositivo: {device}\n")

    for caminho in caminhos:
        texto = inferir(caminho, processor, model, device)
        print(f"  {caminho}  →  {texto}")


if __name__ == "__main__":
    main()
