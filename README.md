# ssd-w2v2-lang

Code + datasets for the paper SYNTHETIC SPEECH DETECTION WITH WAV2VEC 2.0 IN VARIOUS LANGUAGE SETTINGS

URL: [TODO]

## Introduction

Repo contains:
- training/evaluation framework on top of huggingface transformers, pytorch-image-models and it employs pytorch-lightning
- dataset preparation scripts (See: src/data_)
- TTS generation scripts (See: src/data_)

## Installation & Requirements

- create conda env: `conda create --name $CUSTOM_NAME python=3.8 -y`
- activate env: `conda activate $CUSTOM_NAME`
- install PyTorch: e.g. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
- install other requirements: `pip install -r requirements.txt`

## Usage

- train/eval scripts take `.csv` file with at least 2 columns: label column (can contain integers or strings e.g. `"robot"`) and the column containing relative paths to audio files (e.g. `tts/tts_type/audios/100.wav`).
- flag `--audio_root_dir` along with relative path to audio from `.csv` files should form absolute path to every audio file
- scripts have main sources of models: Wav2Vec2 huggingface transformers (`"hft"`, for transfomer models on audio files)
- augmentations
- for any additional info on flags: `python3 train.py --help`

### FT Wav2Vec2 model:

```bash
TRAIN_FILEPATH="/data/fleurs_and_tts/train.csv"
EVAL_FILEPATH="/data/fleurs_and_tts/dev.csv"
AUDIO_ROOT_DIR="/data/audio"

AUGM_CONFIG_PATH="/data/configs/augmentation_config.yaml"

MODEL_NAME_OR_PATH="facebook/wav2vec2-base"
OUTPUT_DIR="outputs/proba_fleurs"


python3 train.py \
    --accelerator "gpu" \
    --gpu "0" \
    --devices 1 \
    \
    --train_filepath $TRAIN_FILEPATH \
    --eval_filepath $EVAL_FILEPATH \
    --audio_filepart_column "audio_relative_path" \
    --labels_column "label" \
    --audio_root_dir $AUDIO_ROOT_DIR \
    --possible_labels "human" "robot" \
    --max_sequence_length 32000 \
    --sample_rate 16000 \
    --multiply_sample_and_cache_random_window \
    --multiply_factor 1 \
    \
    --model_source "hft" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --freeze_feature_encoder \
    \
    --use_augmentations \
    --augmentations_config_path $AUGM_CONFIG_PATH \
    \
    --n_epochs 20 \
    --num_workers 8 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --learning_rate "5e-6" \
    --scheduler_kwargs '{"type": "reduce_on_plateau", "factor": 0.1, "patience": 2, "cooldown": 1, "mode": "max"}' \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 1 \
    --early_stopping_patience 6 \
    --seed 45 \
    --callback_monitor "f1_macro_dev" \
    --callback_mode "max" \
    \
    --output_dir $OUTPUT_DIR \
```

### EVAL Wav2Vec2 model:

```bash
EVAL_FILEPATH="/data/fleurs_and_tts/dev.csv"
AUDIO_ROOT_DIR="/data/audio"

MODEL_NAME_OR_PATH="/data/outputs/model.ckpt"

AUGM_CONFIG_PATH="/data/configs/augmentation_config.yaml"


python evaluate.py \
    --accelerator "gpu" \
    --gpu "1" \
    --devices 1 \
    \
    --eval_filepath $EVAL_FILEPATH \
    --audio_filepart_column "audio_relative_path" \
    --labels_column "label" \
    --audio_root_dir $AUDIO_ROOT_DIR \
    --possible_labels "human" "robot" \
    --max_sequence_length 32000 \
    --sample_rate 16000 \
    --multiply_sample_and_cache_random_window \
    --multiply_factor 1 \
    \
    --model_source "hft" \
    --model_name "facebook/wav2vec2-base" \
    --model_path $MODEL_NAME_OR_PATH \
    \
    --num_workers 8 \
    --eval_batch_size 16 \
    --seed 45 \
    \
    # --use_augmentations \
    # --augmentations_config_path $AUGM_CONFIG_PATH \
```

### Augmentations config:

- Example (`yaml` file):
```yaml
time_augmentations:
  impulse_response:
    p: 0.4
    ir_path: '/data/background_noises/impulse_responses/EchoThiefImpulseResponseLibraryStandardized'
  seven_band_parametric_eq:
    p: 0.2
  biquad_peaking_filter:
    p: 0.2
  air_absorption:
    p: 0.3
  clipping_distortion:
    p: 0.3
  tanh_distortion:
    p: 0.1
  gaussian_noise:
    p: 0.5
    max_amplitude: 0.05
    min_amplitude: 0.01
  gaussian_snr:
    p: 0.1
  background_noise:
    p: 0.6
    sounds_path: '/data/background_noises/long_noises/AmbienceSoundEffectsStandardized'
    sample_rate: 16000
  time_mask:
    p: 0.3

```
- see `AUGMENTATION_NAME_TO_CLASS` map in `src/augmentation.py` on how augmentation name maps to specific augmentation (either from audiomentations or custom)
- in the configuration file you can control the value of any parameter that is implemented in the specific augmentation class