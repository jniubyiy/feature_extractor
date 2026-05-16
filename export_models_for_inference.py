# export_models_for_inference.py
import torch
import os
import glob
from pathlib import Path
from model_Autoencoder import Encoder, Decoder
from model_ParnetCompressor import ParnetCompressor, ParnetDecompressor
from model_ParnetCompressorLevel2 import ParnetCompressorLevel2, ParnetDecompressorLevel2
from config_training_models_Encoder_Decoder import ENCODER_CONFIG, DECODER_CONFIG, IMAGE_SIZE as IMG_SIZE_ENC
from config_training_models_Compressor_Decompressor import COMPRESSOR_CONFIG, DECOMPRESSOR_CONFIG
from config_training_models_Compressor_Decompressor_Level2 import COMPRESSOR_CONFIG as COMP2_CONFIG, DECOMPRESSOR_CONFIG as DECOMP2_CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "encoder": (Encoder, ENCODER_CONFIG),
    "decoder": (Decoder, DECODER_CONFIG),
    "compressor": (ParnetCompressor, COMPRESSOR_CONFIG),
    "decompressor": (ParnetDecompressor, DECOMPRESSOR_CONFIG),
    "compressor_level2": (ParnetCompressorLevel2, COMP2_CONFIG),
    "decompressor_level2": (ParnetDecompressorLevel2, DECOMP2_CONFIG),
}

INPUT_SHAPES = {
    "encoder": (3, IMG_SIZE_ENC, IMG_SIZE_ENC),
    "decoder": (3, IMG_SIZE_ENC, IMG_SIZE_ENC),
    "compressor": (3, IMG_SIZE_ENC, IMG_SIZE_ENC),
    "decompressor": (4, IMG_SIZE_ENC // 2, IMG_SIZE_ENC // 2),
    "compressor_level2": (4, IMG_SIZE_ENC // 2, IMG_SIZE_ENC // 2),
    "decompressor_level2": (5, IMG_SIZE_ENC // 4, IMG_SIZE_ENC // 4),
}

def export_single_model(ckpt_path: Path, output_dir: Path):
    fname = ckpt_path.stem
    parts = fname.split('_epoch')
    if len(parts) != 2:
        print(f"Пропуск {ckpt_path}: неверный формат имени")
        return
    model_name = parts[0]

    if model_name not in MODEL_REGISTRY:
        print(f"Пропуск {ckpt_path}: неизвестная модель '{model_name}'")
        return

    print(f"Экспорт {model_name} из {ckpt_path} ...")
    ModelClass, config = MODEL_REGISTRY[model_name]
    model = ModelClass(**config).to(DEVICE)

    # Загружаем чекпоинт, учитывая возможный формат (полный или только state_dict)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Полный чекпоинт (с оптимизатором и эпохой)
        state_dict = checkpoint["model_state_dict"]
    else:
        # Только state_dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    input_shape = (1,) + INPUT_SHAPES[model_name]
    example_input = torch.randn(*input_shape, device=DEVICE)

    try:
        traced_model = torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"Ошибка трассировки {model_name}: {e}. Пробуем script...")
        try:
            traced_model = torch.jit.script(model)
        except Exception as e2:
            print(f"Не удалось экспортировать {model_name}: {e2}")
            return

    output_path = output_dir / f"{model_name}_inference.pt"
    torch.jit.save(traced_model, str(output_path))
    print(f"Сохранён {output_path}")

def main():
    base_dirs = [
        Path("./models"),
        Path("./models_compressor"),
        Path("./models_compressor_level2"),
    ]

    for base_dir in base_dirs:
        if not base_dir.exists():
            print(f"Папка {base_dir} не найдена, пропуск.")
            continue
        ckpt_files = sorted(glob.glob(str(base_dir / "*_epoch*.pth")))
        for ckpt_path in ckpt_files:
            export_single_model(Path(ckpt_path), base_dir)

if __name__ == "__main__":
    main()