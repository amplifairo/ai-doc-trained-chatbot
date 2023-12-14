# chatbot based on openAI and langchain

AI Based chatbot to use with your own files.


## Installation

Clone the repository :

`git clone https://github.com/amplifairo/ai-doc-trained-chatbot.git`


Navigate to the project directory :

`cd ai-doc-trained-chatbot`


Create a virtual environment:
LINUX
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

MACOS
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```
Modify `constants.py.example` to use your own [OpenAI API key](https://platform.openai.com/account/api-keys), and rename it to `constants.py`.

Place your own data into `data/` folder (for best results use txt files).

## Example usage
Test reading `data/growcentric_ro.txt` file.
```
> streamlit run main.py
```

For embedig into you site just use code from `deploy_template.js` and replace `___IFRAME_SOURCE___` with the address were you've deployed the chat.