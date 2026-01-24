# Copilot AI Test Suite

Комплексный набор тестов для Copilot AI service - OpenAI-совместимого роутера к OpenRouter.

## 📂 Структура

```
tests/
├── __init__.py                 # Package marker
├── conftest.py                 # Shared fixtures and pytest config
├── test_copilot_ai.py          # Main tests (config, models, upstream, API)
├── test_sse_handler.py         # SSE handling tests
├── pytest.ini                  # Pytest configuration
├── requirements-dev.txt        # Development dependencies
├── .coveragerc                 # Coverage configuration
├── .env.test.example           # Example test environment
├── README.md                   # This file
└── TESTING_SUMMARY.md          # Detailed testing summary
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Из корня проекта
pip install -r tests/requirements-dev.txt
```

### Запуск тестов

#### Из корня проекта (рекомендуется)

```bash
# Простой запуск всех тестов
./run_tests.sh

# С coverage
./run_tests.sh -c

# Verbose mode
./run_tests.sh -v

# Параллельный запуск
./run_tests.sh -p

# Конкретный файл
./run_tests.sh -t test_copilot_ai.py

# Комбинирование опций
./run_tests.sh -c -v -p
```

#### Напрямую через pytest

```bash
# Все тесты
pytest tests/

# С verbose
pytest tests/ -v

# С coverage
pytest tests/ --cov=. --cov-config=tests/.coveragerc --cov-report=html

# Конкретный файл
pytest tests/test_copilot_ai.py

# Конкретный класс
pytest tests/test_copilot_ai.py::TestConfig

# Конкретный тест
pytest tests/test_copilot_ai.py::TestConfig::test_from_env_defaults
```

## 📋 Что покрывают тесты

### test_copilot_ai.py (42 теста)

#### TestConfig - Конфигурация
- ✅ Загрузка из environment variables
- ✅ Валидация параметров
- ✅ Кастомные значения
- ✅ Обработка ошибок

#### TestModelInfo - Модели
- ✅ Свойства моделей (max_price, has_tools_support)
- ✅ Преобразование в virtual model

#### TestModelBanList - Система банов
- ✅ Временные баны с TTL
- ✅ Проверка статуса бана
- ✅ Автоматическая очистка expired банов

#### TestModelCache - Кэширование
- ✅ Получение и кэширование моделей
- ✅ Автообновление по таймеру
- ✅ Обработка ошибок upstream

#### TestModelSelector - Выбор моделей
- ✅ Фильтрация (context, price, tools, bans)
- ✅ Приоритизация моделей
- ✅ Сортировка кандидатов

#### TestUpstreamClient - Upstream API
- ✅ Формирование headers
- ✅ Chat completion (streaming/non-streaming)
- ✅ Чтение error snippets
- ✅ Валидация ответов

#### TestAPIEndpoints - API
- ✅ Health check
- ✅ Models listing
- ✅ Chat completions
- ✅ Валидация запросов
- ✅ Обработка ошибок

#### TestIntegration - Интеграция
- ✅ Workflow выбора модели
- ✅ Workflow банов
- ✅ Нормализация виртуальных моделей

### test_sse_handler.py (14 тестов)

#### TestSSEHelpers - Helper функции
- ✅ Определение SSE activity lines
- ✅ Детектирование DONE events
- ✅ Распознавание inline tool calls
- ✅ Извлечение content fragments

#### TestSSEValidator - Валидация SSE
- ✅ Peek первого SSE chunk
- ✅ Prebuffering стрима
- ✅ Детектирование inline tool calls
- ✅ Обработка early EOF
- ✅ Автоматический ban проблемных моделей

#### TestSSEStreamer - Streaming
- ✅ Стриминг с watchdog
- ✅ Обработка stalled connections
- ✅ Генерация error responses

## 📊 Статистика

- **Всего тестов:** 56
- **Покрытие:** ~85%
- **Типы:** Unit (84%), Integration (16%), Async (39%)

## 🔧 Конфигурация

### pytest.ini
- Asyncio mode: auto
- Verbose output по умолчанию
- Таймаут: 30 секунд
- Маркеры для организации тестов

### .coveragerc
- Исключение тестовых файлов
- Настройки отчетов
- HTML отчеты в `htmlcov/`

## 🎯 Fixtures

### conftest.py

```python
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Автоматическая настройка окружения"""
    pass

@pytest.fixture(scope="session")
def project_root_path():
    """Путь к корню проекта"""
    pass
```

### test_copilot_ai.py

```python
@pytest.fixture
def test_config():
    """Тестовая конфигурация"""
    pass

@pytest.fixture
def sample_models():
    """Примеры моделей"""
    pass

@pytest.fixture
def mock_httpx_client():
    """Mock HTTP клиент"""
    pass

@pytest.fixture
def client():
    """FastAPI test client"""
    pass
```

## 📝 Написание новых тестов

### Базовая структура

```python
class TestNewFeature:
    """Test new feature."""

    def test_something(self):
        """Test description."""
        # Arrange
        data = ...

        # Act
        result = function(data)

        # Assert
        assert result == expected
```

### Async тесты

```python
@pytest.mark.asyncio
async def test_async_feature():
    """Test async feature."""
    result = await async_function()
    assert result == expected
```

### Использование fixtures

```python
def test_with_fixture(test_config, sample_models):
    """Test using fixtures."""
    result = process(test_config, sample_models)
    assert result is not None
```

### Мокирование

```python
from unittest.mock import AsyncMock, MagicMock, patch

def test_with_mock():
    """Test with mocking."""
    mock_client = MagicMock()
    mock_client.method = AsyncMock(return_value="result")

    result = await function(mock_client)
    assert result == "result"
```

## 🏷️ Маркеры

Используйте маркеры для организации тестов:

```python
@pytest.mark.unit
def test_unit():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.requires_api_key
def test_with_api():
    pass
```

Запуск по маркерам:
```bash
pytest tests/ -m unit           # Только unit тесты
pytest tests/ -m "not slow"     # Исключить медленные
```

## 🐛 Troubleshooting

### ImportError при запуске тестов

**Проблема:** Тесты не могут импортировать модули проекта.

**Решение:** `conftest.py` автоматически добавляет родительский каталог в `sys.path`. Убедитесь, что запускаете тесты из корня проекта.

### Async тесты не работают

**Проблема:** `RuntimeError: no running event loop`

**Решение:**

**Проблема:** `RuntimeError: no running event loop`

**Решение:**
- Убедитесь что установлен `pytest-asyncio`
- Проверьте что используется `@pytest.mark.asyncio`
- В `pytest.ini` должно быть `asyncio_mode = auto`

### Coverage не работает

**Проблема:** Coverage не находит исходные файлы.

**Решение:**
```bash
# Используйте правильный путь к конфигу
pytest tests/ --cov=. --cov-config=tests/.coveragerc
```

### Медленные тесты

**Решение:**
```bash
# Параллельный запуск
pip install pytest-xdist
pytest tests/ -n auto
```

### 🔍 Отладка с req_id

Все основные тесты теперь корректно работают с `req_id` в логах:

```python
# В любом месте теста можно включить интеграционное логирование
def test_with_logging(self, caplog):
    with caplog.at_level(logging.DEBUG):
        # Ваш тестовый код
        pass

    # Проверить наличие req_id в логах
    assert any("req_id=" in record.message for record in caplog.records)
```

## 📚 Дополнительная документация

- [TESTING_SUMMARY.md](TESTING_SUMMARY.md) - Детальная сводка по тестированию
- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 🔄 CI/CD

### GitHub Actions пример

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r tests/requirements-dev.txt
      - run: ./run_tests.sh -c
      - uses: codecov/codecov-action@v3
```

## 📄 Лицензия

Тесты распространяются под той же лицензией, что и основной проект.
