# Streamlit Usage
## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gsoc24-zebinhuang.streamlit.app/)

This repository is deployed on a personal Streamlit community cloud account.

Note that Streamlit Community Cloud runs with the working directory set to /mount/src/gsoc-streamlit by default.

### Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:

1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

## Run it locally

```sh
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit/main.py
```

- The model training process may be slow due to the limitations of Streamlit Community Cloud.
- The model used for testing is large and is hosted using git lfs. For more details, you can also check the [Hugging Face models](https://huggingface.co/zebin-huang/gsoc2024-ZebinHuang/tree/main).
