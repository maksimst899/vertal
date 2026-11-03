# vertal
```text
Это мой ии агент VERTAL
V — Versatile (Многофункциональный)
E — Experimental (Экспериментальный)
R — Reasoning (Мышление) |
T — Tactical (Тактический)
A — Autonomous (Автономный)
L — Logic (Логика)
```
Для запуска этого Python-приложения с графическим интерфейсом потребуются PyQt5 и Ollama.

После установки Ollama программа подключается к локальному серверу Ollama по адресу http://localhost:11434.

Установка:
pip install ollama PyQt5

ollama pull gpt-oss-safeguard:20b

В коде есть обработка ошибки, где указан путь к моделям — меняйте его под себя: OLLAMA_MODELS=manifests\registry.ollama.ai\library

Запустите сервер:
ollama serve
