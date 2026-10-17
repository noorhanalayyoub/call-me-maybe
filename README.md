*this activity has been created as part of the 42 curriculum by nalayyou*

# Description


## concepts:
-  AI: broad term for a sub-field of computer science that attempts to create algorithms for computers to learn, reason, and do other human-like things
-  ML: finding a function f that maps inputs x to outputs y, by using data instead of explicit instructions.
-  LLMs: or large language models, kind of AI thats based on deep learning that tries to predict the next word(token) based on what came before it.
-  Tokens: the smaller units that your prompt gets transformed into so that the models understands it.
-  Embedding: representing each token's meaning and its relation to other tokens using a vector. in simpler words taking context into consideration when trying to make sense out of a token.
-  Logits: represents the likelihood of each token appearing next, using a score for each token in the vocabulary, a higher score means a higher likelihood. 
# Instructions

# Algorithm explanation

# Design decisions

# Performance analysis

# Challenges faced

# Testing strategy

# Example usage

## uv package manager
- its an extremely fast python package and project manager written in Rust
- A single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv, and more.
- its much faster than pip
## what makes uv a better package manager
- much faster
- can maintain a project lockfile
- creates a virutal environemnt automatically
- can isntall and manage python versions

## try uv

1 - step 1: create a project
  ```
    uv init my-project
    cd my-project
  ```
2 - step 2: add a dependency:
  ```
    uv add pydantic
  ```

3 - step 3: match the venv's dependencies to your project's:
  ```
    uv sync
  ```
4 - step 4: run your script using the project's environment:
  ```
    uv run main.py
  ```

# Resources
- [what are LLMs - Video](https://www.youtube.com/watch?v=UgvrrHc5BRY)
- [what is deep learning - Video](https://www.youtube.com/watch?v=yAzPUABBkg0)
- [what does the number of parameters in LLMs indicate - Video](https://www.youtube.com/watch?v=dPalBLlA5yQ)
- [Tokenization - Video](https://www.youtube.com/watch?v=bNjVxUDZQfM)
- [Logits in LLMs - Video](https://www.youtube.com/watch?v=N6I1-RrbBmM)
- [Command Line args parsing in python](https://docs.python.org/3/howto/argparse.html#argparse-tutorial)
- [JSON syntax](https://www.w3schools.com/js/js_json_syntax.asp)
