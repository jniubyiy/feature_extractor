# export_models_for_inference.py
"""
Экспорт всех обученных моделей в TorchScript (.pt) для инференса.
Поддерживаются модели из:
  - ./models/                  : Encoder, Decoder
  - ./models_compressor/       : ParnetCompressor, ParnetDecompressor
  - ./models_vae_wrapper/      : StochasticEncoder, StochasticDecoder, VAEWrapper

StochasticEncoder экспортируется с фиксированными параметрами шума
(NOISE_RANGE и STOCHASTIC_STRENGTH), чтобы инференс не требовал внешних конфигураций.
"""
import torch
import os
import glob
from pathlib import Path

# Архитектуры
from model_Autoencoder import Encoder, Decoder
from model_ParnetCompressor import ParnetCompressor, ParnetDecompressor
from model_VAEWrapper import (
    StochasticEncoder, StochasticDecoder, VAEWrapper,
    reparameterize
)

# Конфигурации
from config_training_models_Encoder_Decoder import (
    ENCODER_CONFIG, DECODER_CONFIG, IMAGE_SIZE as IMG_SIZE_ENC
)
from config_training_models_Compressor_Decompressor import (
    COMPRESSOR_CONFIG, DECOMPRESSOR_CONFIG
)
from config_training_VAEWrapper import (
    STOCHASTIC_ENCODER_CONFIG,
    STOCHASTIC_DECODER_CONFIG,
    IMAGE_SIZE as IMG_SIZE_VAE,
    NOISE_RANGE,
    STOCHASTIC_STRENGTH
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# Обёртка для StochasticEncoder: вшиваем параметры шума в TorchScript
# ----------------------------------------------------------------------
class StochasticEncoderInference(torch.nn.Module):
    """Экспортирует StochasticEncoder + фиксированную репараметризацию с равномерным шумом."""
    def __init__(self, encoder, noise_range, strength):
        super().__init__()
        self.encoder = encoder
        self.noise_range = noise_range
        self.strength = strength

    def forward(self, x):
        mu, noise_seed = self.encoder(x)
        z, _, _ = reparameterize(mu, self.noise_range, self.strength, noise_seed)
        return z, noise_seed

# ----------------------------------------------------------------------
# Сопоставление директорий и моделей
# ----------------------------------------------------------------------
DIR_MODEL_MAP = {
    "./models": {
        "encoder": (Encoder, ENCODER_CONFIG, (3, IMG_SIZE_ENC, IMG_SIZE_ENC)),
        "decoder": (Decoder, DECODER_CONFIG, (3, IMG_SIZE_ENC, IMG_SIZE_ENC)),
    },
    "./models_compressor": {
        "compressor": (ParnetCompressor, COMPRESSOR_CONFIG, (3, IMG_SIZE_ENC, IMG_SIZE_ENC)),
        "decompressor": (ParnetDecompressor, DECOMPRESSOR_CONFIG, (4, IMG_SIZE_ENC // 2, IMG_SIZE_ENC // 2)),
    },
    "./models_vae_wrapper": {
        "encoder": (
            StochasticEncoder,
            STOCHASTIC_ENCODER_CONFIG,
            (STOCHASTIC_ENCODER_CONFIG["compressed_channels"], IMG_SIZE_VAE // 2, IMG_SIZE_VAE // 2),
        ),
        "decoder": (
            StochasticDecoder,
            STOCHASTIC_DECODER_CONFIG,
            (2 * STOCHASTIC_DECODER_CONFIG["stochastic_parnet_dim"], IMG_SIZE_VAE // 2, IMG_SIZE_VAE // 2),
        ),
        "vae_wrapper": (
            VAEWrapper,
            {
                "compressed_channels": STOCHASTIC_ENCODER_CONFIG["compressed_channels"],
                "stochastic_parnet_dim": STOCHASTIC_ENCODER_CONFIG["stochastic_parnet_dim"],
                "hidden_dim": STOCHASTIC_ENCODER_CONFIG["hidden_dim"],
            },
            (STOCHASTIC_ENCODER_CONFIG["compressed_channels"], IMG_SIZE_VAE // 2, IMG_SIZE_VAE // 2),
        ),
    },
}

# ----------------------------------------------------------------------
# Экспорт одной модели
# ----------------------------------------------------------------------
def export_single_model(ckpt_path: Path, output_dir: Path,
                        model_cls, config, input_shape, model_name,
                        encoder_inference_params=None):
    """
    Загружает веса, создаёт модель (возможно, с обёрткой) и экспортирует в TorchScript.

    encoder_inference_params: (noise_range, strength) только для StochasticEncoder.
    """
    print(f"Экспорт {model_name} из {ckpt_path} ...")
    base_model = model_cls(**config).to(DEVICE)

    # Загрузка весов
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Убираем префикс _orig_mod., если модель была сохранена после torch.compile
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    base_model.load_state_dict(state_dict)
    base_model.eval()

    # Определяем, какую модель будем трассировать
    if encoder_inference_params is not None:
        # StochasticEncoder -> обёртка с фиксированными параметрами шума
        noise_range_val, strength_val = encoder_inference_params
        model_to_trace = StochasticEncoderInference(base_model, noise_range_val, strength_val).to(DEVICE)
        example_input = torch.randn(1, *input_shape, device=DEVICE)
    elif isinstance(base_model, StochasticDecoder):
        # Для StochasticDecoder нужна обёртка, принимающая конкатенацию [z, noise_seed]
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, combined):
                half = combined.shape[1] // 2
                z = combined[:, :half, :, :]
                noise_seed = combined[:, half:, :, :]
                return self.decoder(z, noise_seed)

        model_to_trace = DecoderWrapper(base_model).to(DEVICE)
        example_input = torch.randn(1, *input_shape, device=DEVICE)
    else:
        # Обычная модель
        model_to_trace = base_model
        example_input = torch.randn(1, *input_shape, device=DEVICE)

    # Трассировка (с фолбэком на script)
    try:
        traced = torch.jit.trace(model_to_trace, example_input)
    except Exception as e:
        print(f"Ошибка трассировки {model_name}: {e}. Пробуем script...")
        try:
            traced = torch.jit.script(model_to_trace)
        except Exception as e2:
            print(f"Не удалось экспортировать {model_name}: {e2}")
            return

    output_path = output_dir / f"{model_name}_inference.pt"
    torch.jit.save(traced, str(output_path))
    print(f"Сохранён {output_path}")

# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------
def main():
    for base_dir_str, model_map in DIR_MODEL_MAP.items():
        base_dir = Path(base_dir_str)
        if not base_dir.exists():
            print(f"Папка {base_dir} не найдена, пропуск.")
            continue

        ckpt_files = sorted(glob.glob(str(base_dir / "*_epoch*.pth")))
        if not ckpt_files:
            print(f"В папке {base_dir} нет чекпоинтов формата *_epoch*.pth, пропуск.")
            continue

        for ckpt_path_str in ckpt_files:
            ckpt_path = Path(ckpt_path_str)
            fname = ckpt_path.stem
            parts = fname.split('_epoch')
            if len(parts) != 2:
                print(f"Пропуск {ckpt_path}: неверный формат имени (ожидалось имя_epochN)")
                continue
            model_name = parts[0]  # "encoder", "decoder", "compressor" и т.д.

            if model_name not in model_map:
                print(f"Пропуск {ckpt_path}: неизвестная модель '{model_name}' в {base_dir}")
                continue

            model_cls, config, input_shape = model_map[model_name]

            # Особый случай: encoder из models_vae_wrapper использует фиксированные параметры шума
            if base_dir_str == "./models_vae_wrapper" and model_name == "encoder":
                params = (NOISE_RANGE, STOCHASTIC_STRENGTH)
                export_single_model(ckpt_path, base_dir, model_cls, config, input_shape,
                                    model_name, encoder_inference_params=params)
            else:
                export_single_model(ckpt_path, base_dir, model_cls, config, input_shape,
                                    model_name)

    print("Экспорт завершён.")

if __name__ == "__main__":
    main()