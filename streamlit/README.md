## Streamlit Usage

Start Streamlit local pages by navigating to the `data_parsing` directory and running the main script:

```bash
streamlit run streamlit/main.py
```

Note that Streamlit Community Cloud runs with the working directory set to /mount/src/gsoc-streamlit by default.

This repository is deployed on a personal Streamlit community cloud account. You can directly access the page here: https://gsoc24-zebinhuang.streamlit.app/

When deploying the app, remember to include the OpenAI API key:

```bash
openai_api_key = "your_api_key"
```

For more details on setting this up, refer to [Streamlit's discussion on setting OpenAI API keys](https://discuss.streamlit.io/t/struggling-with-setting-openai-api-using-streamlit-secrets/37959).


Deployment Notes
- The model training process may be slow due to the limitations of Streamlit Community Cloud.
- The model used for testing is large and cannot be hosted directly on the GitHub repository. Currently, this model is not included in the Streamlit deployment.
