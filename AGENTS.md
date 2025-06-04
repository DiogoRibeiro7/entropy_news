# AGENTS Instructions

Bem-vindo ao repositório **Entropy News**.

Este ficheiro define orientações para futuras alterações. Todas as modificações no árvore a partir deste diretório devem seguir estas regras.

## Estilo de código

- Utilize Python 3.10 ou superior com *type hints* e *docstrings* em todas as funções.
- Prefira `snake_case` para variáveis e funções.
- Comentários e mensagens devem ser breves e, quando possível, em inglês.

## Funcionalidades recomendadas

- Os scripts `main.py` e `main_forecast.py` já suportam `argparse` para permitir parametrização via linha de comando.
- O carregamento do GloVe detecta automaticamente a dimensão das *embeddings* (podendo ser sobreposta com argumento).
- A classe `EntropyLSTM` permite configurar `num_layers` e `dropout`.
- Testes unitários para `EntropyCalculator` e `NewsModelUpdateCalculator` encontram-se em `tests/`.

Qualquer nova funcionalidade deve manter compatibilidade com estas extensões.

## Testes

- Ao alterar código Python, execute `pytest -q` a partir da raiz do repositório e assegure-se de que todos os testes passam.
- Se novos comportamentos forem adicionados, inclua testes correspondentes dentro de `tests/`.

## Pull Requests

- Mensagens de *commit* devem ser concisas (ex.: `feat: add cli support`).
- O corpo do PR deve descrever resumidamente as alterações e referir se os testes passaram.

