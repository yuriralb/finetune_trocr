"""
Inferência com o modelo TrOCR fine-tunado.

USO:
    # Uma imagem:
    python inferir.py caminho/para/imagem.jpg

    # Várias imagens:
    python inferir.py imagem1.jpg imagem2.png imagem3.JPEG

    # Todas as imagens de uma pasta:
    python inferir.py frases/*.JPEG
"""

import sys
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    RobertaTokenizer,
)

MODEL_DIR = "./trocr-minha-letra"


def carregar_modelo(model_dir):
    """Carrega o modelo e o processor salvos."""
    image_processor = ViTImageProcessor.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return processor, model, device


def inferir(image_path, processor, model, device):
    """Recebe o caminho de uma imagem e retorna o texto previsto."""
    imagem = Image.open(image_path).convert("RGB")
    pixel_values = processor(imagem, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    texto = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return texto


def main():
    if len(sys.argv) < 2:
        print("Uso: python inferir.py <imagem1> [imagem2] [imagem3] ...")
        print("Exemplo: python inferir.py frases/image1.JPEG")
        sys.exit(1)

    caminhos = sys.argv[1:]

    print("Carregando modelo...")
    processor, model, device = carregar_modelo(MODEL_DIR)
    print(f"Modelo carregado no dispositivo: {device}\n")

    for caminho in caminhos:
        texto = inferir(caminho, processor, model, device)
        print(f"  {caminho}  →  {texto}")


if __name__ == "__main__":
    main()
