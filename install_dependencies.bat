@echo off
echo Установка зависимостей проекта feature_extractor

:: Установка PyTorch с CUDA 11.8
echo.
echo [1/2] Установка PyTorch для CUDA 11.8...
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

:: Установка остальных пакетов из requirements.txt
echo.
echo [2/2] Установка остальных зависимостей...
pip install -r requirements.txt

echo.
echo Готово! Все зависимости установлены.
pause