# Models Directory

This directory is designated for local model configuration, offline model weights, or cached artifacts if applicable.

By default, the application uses Hugging Face's pretrained `google/flan-t5-base` model downloaded dynamically into the standard Hugging Face cache (`~/.cache/huggingface/hub/`).

If running offline or with pre-downloaded weights, point the `NLG_MODEL_NAME` environment variable to the path of your local model folder inside this directory.
