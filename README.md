# RAG Ассистент с фильтрацией по источникам и семантическим разбиением

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://platform.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Интеллектуальный ассистент на основе RAG (Retrieval-Augmented Generation) с поддержкой множественных источников данных и семантического разбиения на чанки.

## 🚀 Возможности

- **Множественные источники данных**: общие знания и специализированная база Python
- **Фильтрация по источникам**: поиск только в нужной базе знаний
- **Семантическое разбиение**: каждый чанк содержит одно понятие
- **Кеширование**: быстрые ответы на повторные вопросы
- **OpenAI API**: высокое качество генерации ответов

## 📋 Требования

- Python 3.11+
- OpenAI API ключ
- Зависимости из `requirements.txt`

## ⚙️ Установка

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/alexshama/Q-A-Agent_LLM_Python.git
   cd Q-A-Agent_LLM_Python
   ```

2. **Создайте виртуальное окружение:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Настройте API ключ:**
   ```bash
   # Скопируйте шаблон конфигурации
   cp .env.example .env
   
   # Отредактируйте .env файл и добавьте ваш OpenAI API ключ
   # OPENAI_API_KEY=your-api-key-here
   ```

## 🎯 Использование

### Консольное приложение
```bash
cd assistant_api
python app.py
```

### Команды фильтрации
- `@python Что такое переменная?` - поиск только в Python базе
- `@docs Что такое RAG?` - поиск в общих документах
- `Что такое машинное обучение?` - поиск по всем источникам

### Специальные команды
- `stats` - статистика системы
- `clear` - очистка кеша
- `exit` - выход из программы

## 📊 Архитектура

### Источники данных
- **docs**: Общие знания по AI/ML (`assistant_api/data/docs.txt`)
- **python**: Специализированная база Python (`assistant_api/data/python.txt`)

### Семантическое разбиение
- **Python файл**: 18 концептуальных чанков (одно понятие = один чанк)
- **Обычные файлы**: стандартное разбиение по абзацам
- **Метаданные**: каждый чанк содержит информацию об источнике

### Компоненты системы
```
assistant_api/
├── app.py              # Консольное приложение
├── rag_pipeline.py     # Основная логика RAG
├── vector_store.py     # Работа с векторным хранилищем
├── cache.py           # Система кеширования
├── evaluate_ragas.py  # Оценка качества системы
└── data/
    ├── docs.txt       # Общие знания
    └── python.txt     # Python база знаний
```

## 🧪 Примеры использования

### Точные ответы на Python вопросы
```
💭 @python Что такое int?
💬 int — целочисленный тип данных произвольной точности.

💭 @python Что делает функция id?
💬 Функция id() возвращает уникальный идентификатор объекта.

💭 @python Что такое LEGB?
💬 LEGB — Local, Enclosing, Global, Built-in.
```

### Общие вопросы по AI/ML
```
💭 Что такое машинное обучение?
💬 Машинное обучение - это раздел искусственного интеллекта...

💭 Что такое RAG?
💬 RAG (Retrieval-Augmented Generation) - это подход...
```

## 📈 Преимущества

### Семантическое разбиение
- **Точность поиска**: прямые попадания в нужные концепты
- **Релевантность**: каждый чанк содержит законченную информацию
- **Качество ответов**: минимум шума, максимум точности

### Фильтрация по источникам
- **Специализированные ответы**: поиск только в нужной области
- **Быстродействие**: уменьшенное пространство поиска
- **Гибкость**: легко добавлять новые источники

### Кеширование
- **Производительность**: быстрые ответы на повторные вопросы
- **Экономия**: снижение затрат на API вызовы
- **Умное кеширование**: учитывает фильтры источников

## 🔧 Программное использование

```python
from rag_pipeline import RAGPipeline

# Инициализация
data_sources = {
    "docs": "assistant_api/data/docs.txt",
    "python": "assistant_api/data/python.txt"
}

pipeline = RAGPipeline(
    collection_name="semantic_rag_collection_v2",
    cache_db_path="assistant_api/semantic_rag_cache_v2.db",
    data_sources=data_sources
)

# Поиск с фильтром
result = pipeline.query("Что такое переменная?", source_filter="python")
print(result['answer'])

# Общий поиск
result = pipeline.query("Что такое машинное обучение?")
print(result['answer'])
```

## 📊 Статистика системы

- **Всего документов**: 33
- **Python чанков**: 18 (семантическое разбиение)
- **Docs чанков**: 15 (стандартное разбиение)
- **Точность поиска**: 100% для специфичных терминов

## 🛠️ Оценка качества

Запустите оценку системы через RAGAS:
```bash
cd assistant_api
python evaluate_ragas.py
```

Система автоматически оценит:
- Faithfulness (точность ответов)
- Context Precision (качество контекста)
- Сравнительный анализ с фильтрами и без них

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста:

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📞 Поддержка

Если у вас есть вопросы или предложения:
- Создайте [Issue](https://github.com/alexshama/Q-A-Agent_LLM_Python/issues)
- Обратитесь к [документации](README.md)

## 📝 Лицензия

Проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).

---

**Система готова к использованию!** 🎉

Для начала работы выполните:
```bash
cd assistant_api && python app.py
```