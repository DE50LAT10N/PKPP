import os
import sys
import warnings
import pytest

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Фильтруем все предупреждения pymorphy2
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", module="pymorphy2.*")

# Можно добавить другие настройки pytest здесь
def pytest_configure(config):
    """Настройка pytest при запуске"""
    # Добавляем дополнительные фильтры предупреждений
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:pymorphy2.*:"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:setuptools.*:"
    ) 