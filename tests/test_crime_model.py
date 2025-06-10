import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from models.crime_model import CrimeModel, CrimeDataset
import torch

@pytest.fixture
def sample_texts_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'text': [
            'В Москве произошло ограбление банка с использованием пистолета.',
            'В Санкт-Петербурге совершена кража с применением ножа.',
            'В Казани произошло нападение с использованием биты.'
        ]
    })

@pytest.fixture
def crime_model():
    """Create a CrimeModel instance for testing"""
    return CrimeModel()

def test_crime_dataset_initialization(sample_texts_df):
    """Test CrimeDataset initialization"""
    # Test with default text column
    dataset = CrimeDataset(sample_texts_df, Mock())
    assert len(dataset) == 3
    assert dataset.texts == sample_texts_df['text'].values.tolist()

    # Test with specified text column
    dataset = CrimeDataset(sample_texts_df, Mock(), text_column='text')
    assert len(dataset) == 3
    assert dataset.texts == sample_texts_df['text'].values.tolist()

def test_crime_dataset_validation():
    """Test CrimeDataset validation"""
    # Test with non-DataFrame input
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        CrimeDataset("not a dataframe", Mock())

    # Test with empty DataFrame
    with pytest.raises(ValueError, match="DataFrame is empty"):
        CrimeDataset(pd.DataFrame(), Mock())

    # Test with non-existent column
    with pytest.raises(ValueError, match="Column 'nonexistent' not found in DataFrame"):
        CrimeDataset(pd.DataFrame({'text': ['test']}), Mock(), text_column='nonexistent')

    # Test with all NA values
    with pytest.raises(ValueError, match="Column 'text' contains only empty values"):
        CrimeDataset(pd.DataFrame({'text': [None, None]}), Mock())

@patch('models.crime_model.AutoTokenizer')
@patch('models.crime_model.AutoModel')
def test_load_data(mock_model, mock_tokenizer, crime_model, sample_texts_df, tmp_path):
    """Test load_data method"""
    # Create a temporary Excel file
    file_path = tmp_path / "test_data.xlsx"
    sample_texts_df.to_excel(file_path, index=False)

    # Mock the model and tokenizer
    mock_model.from_pretrained.return_value = Mock()
    mock_tokenizer.from_pretrained.return_value = Mock()

    # Mock the embeddings and clustering
    with patch.object(crime_model, '_get_embeddings') as mock_get_embeddings, \
         patch.object(crime_model, '_cluster_texts_with_optics') as mock_cluster:
        
        mock_get_embeddings.return_value = np.random.rand(3, 768)
        mock_cluster.return_value = (
            np.array([0, 0, 1]),  # cluster labels
            [(0, 2), (1, 1)]     # valid clusters
        )

        result = crime_model.load_data(str(file_path))

        assert 'texts' in result
        assert 'embeddings' in result
        assert 'cluster_labels' in result
        assert 'clusters' in result
        assert len(result['texts']) == 3

def test_get_cluster_texts(crime_model):
    """Test get_cluster_texts method"""
    crime_model.texts = ['text1', 'text2', 'text3']
    crime_model.cluster_labels = [0, 0, 1]

    # Test getting texts for cluster 0
    cluster_0_texts = crime_model.get_cluster_texts(0)
    assert len(cluster_0_texts) == 2
    assert 'text1' in cluster_0_texts
    assert 'text2' in cluster_0_texts

    # Test getting texts for cluster 1
    cluster_1_texts = crime_model.get_cluster_texts(1)
    assert len(cluster_1_texts) == 1
    assert 'text3' in cluster_1_texts

def test_add_and_get_generated_texts(crime_model):
    """Test add_generated_text and get_generated_texts methods"""
    # Test adding generated text
    crime_model.add_generated_text(0, "Generated text 1")
    crime_model.add_generated_text(0, "Generated text 2")
    crime_model.add_generated_text(1, "Generated text 3")

    # Test getting generated texts
    cluster_0_texts = crime_model.get_generated_texts(0)
    assert len(cluster_0_texts) == 2
    assert "Generated text 1" in cluster_0_texts
    assert "Generated text 2" in cluster_0_texts

    cluster_1_texts = crime_model.get_generated_texts(1)
    assert len(cluster_1_texts) == 1
    assert "Generated text 3" in cluster_1_texts

    # Test getting texts for non-existent cluster
    assert crime_model.get_generated_texts(2) == []

def test_extract_entities(crime_model):
    """Test _extract_entities method"""
    text = "Иван Иванов совершил ограбление в Москве с использованием пистолета."
    
    # Test extracting specific entity types
    entities = crime_model._extract_entities(text, ['PER', 'LOC'])
    assert 'PER' in entities
    assert 'LOC' in entities
    assert 'Иван Иванов' in entities['PER']
    assert 'Москве' in entities['LOC']

    # Test with empty text
    assert crime_model._extract_entities("", ['PER', 'LOC']) == {'PER': [], 'LOC': []}

    # Test with non-string input
    assert crime_model._extract_entities(None, ['PER', 'LOC']) == {'PER': [], 'LOC': []}

def test_extract_weapons(crime_model):
    """Test _extract_weapons method"""
    # Test with various weapons
    text = "Преступник использовал нож и пистолет для ограбления."
    weapons = crime_model._extract_weapons(text)
    assert 'нож' in weapons
    assert 'пистолет' in weapons

    # Test with no weapons
    text = "Произошло ограбление без применения оружия."
    weapons = crime_model._extract_weapons(text)
    assert len(weapons) == 0

    # Test with empty text
    assert crime_model._extract_weapons("") == []

    # Test with non-string input
    assert crime_model._extract_weapons(None) == []

def test_extract_entities_edge_cases(crime_model):
    """Test _extract_entities method with edge cases"""
    # Test with empty entity types
    assert crime_model._extract_entities("text", []) == {}
    
    # Test with non-existent entity types
    assert crime_model._extract_entities("text", ['NONEXISTENT']) == {'NONEXISTENT': []}
    
    # Test with multiple entities of same type
    text = "Иван и Петр совершили ограбление в Москве и Санкт-Петербурге."
    entities = crime_model._extract_entities(text, ['PER', 'LOC'])
    assert len(entities['PER']) == 2
    assert len(entities['LOC']) == 2
    assert 'Иван' in entities['PER']
    assert 'Петр' in entities['PER']
    assert 'Москве' in entities['LOC']
    assert 'Санкт-Петербурге' in entities['LOC']

def test_extract_weapons_edge_cases(crime_model):
    """Test _extract_weapons method with edge cases"""
    # Test with multiple weapons
    text = "Преступник использовал нож, пистолет и биту для ограбления."
    weapons = crime_model._extract_weapons(text)
    assert len(weapons) == 3
    assert 'нож' in weapons
    assert 'пистолет' in weapons
    assert 'бита' in weapons
    
    # Test with weapons in different cases
    text = "Использовал ПИСТОЛЕТ и Нож для ограбления."
    weapons = crime_model._extract_weapons(text)
    assert len(weapons) == 2
    assert 'пистолет' in weapons
    assert 'нож' in weapons
    
    # Test with weapons in different forms
    text = "Использовал пистолеты и ножи для ограбления."
    weapons = crime_model._extract_weapons(text)
    assert len(weapons) == 2
    assert 'пистолет' in weapons
    assert 'нож' in weapons

def test_get_cluster_texts_edge_cases(crime_model):
    """Test get_cluster_texts method with edge cases"""
    # Test with empty data
    crime_model.texts = []
    crime_model.cluster_labels = []
    assert crime_model.get_cluster_texts(0) == []
    
    # Test with non-existent cluster
    crime_model.texts = ['text1', 'text2']
    crime_model.cluster_labels = [0, 0]
    assert crime_model.get_cluster_texts(1) == []
    
    # Test with single text in cluster
    crime_model.texts = ['text1']
    crime_model.cluster_labels = [0]
    assert crime_model.get_cluster_texts(0) == ['text1']

def test_add_generated_text_edge_cases(crime_model):
    """Test add_generated_text method with edge cases"""
    # Test adding text to non-existent cluster
    crime_model.add_generated_text(0, "text1")
    assert crime_model.get_generated_texts(0) == ["text1"]
    
    # Test adding multiple texts to same cluster
    crime_model.add_generated_text(0, "text2")
    assert len(crime_model.get_generated_texts(0)) == 2
    assert "text1" in crime_model.get_generated_texts(0)
    assert "text2" in crime_model.get_generated_texts(0)
    
    # Test adding text to different clusters
    crime_model.add_generated_text(1, "text3")
    assert crime_model.get_generated_texts(1) == ["text3"]
    assert len(crime_model.get_generated_texts(0)) == 2

def test_crime_dataset_edge_cases(sample_texts_df):
    """Test CrimeDataset with edge cases"""
    # Test with single row DataFrame
    single_row_df = pd.DataFrame({'text': ['single text']})
    dataset = CrimeDataset(single_row_df, Mock())
    assert len(dataset) == 1
    assert dataset.texts == ['single text']
    
    # Test with DataFrame containing NA values
    df_with_na = pd.DataFrame({
        'text': [None, None, None]  # All values are NA
    })
    with pytest.raises(ValueError, match="Column 'text' contains only empty values"):
        CrimeDataset(df_with_na, Mock())
    
    # Test with DataFrame containing empty strings
    df_with_empty = pd.DataFrame({
        'text': ['', '', '']  # All values are empty
    })
    with pytest.raises(ValueError, match="Column 'text' contains only empty values"):
        CrimeDataset(df_with_empty, Mock())

def test_export_to_excel(crime_model, tmp_path):
    """Test export_to_excel method"""
    # Setup test data
    crime_model.texts = ['text1', 'text2', 'text3']
    crime_model.cluster_labels = [0, 0, 1]
    crime_model.clusters = [(0, 2), (1, 1)]
    crime_model.generated_texts = {
        0: ['generated1', 'generated2'],
        1: ['generated3']
    }
    
    # Мокаем методы извлечения сущностей и оружия, чтобы не зависеть от natasha
    with patch.object(crime_model, '_extract_entities') as mock_extract_entities, \
         patch.object(crime_model, '_extract_weapons') as mock_extract_weapons, \
         patch('pandas.DataFrame.to_excel', autospec=True) as mock_to_excel:
        
        mock_extract_entities.return_value = {
            'LOC': ['Москва'],
            'PER': ['Иван'],
            'PRODUCT': ['пистолет']
        }
        mock_extract_weapons.return_value = ['пистолет']

        file_path = tmp_path / "export_test.xlsx"
        crime_model.export_to_excel(str(file_path))
        
        # Проверяем, что to_excel был вызван с правильными аргументами
        mock_to_excel.assert_called_once()
        args, kwargs = mock_to_excel.call_args
        assert isinstance(args[0], pd.DataFrame)  # Первый аргумент - DataFrame
        assert args[1] == str(file_path)  # Второй аргумент - путь к файлу
        assert not kwargs.get('index', False)  # Проверяем, что index=False
        
        # Проверяем содержимое DataFrame
        df = args[0]
        assert 'Cluster' in df.columns
        assert 'Text' in df.columns
        assert 'Weapons' in df.columns
        assert 'Locations' in df.columns
        assert 'Participants' in df.columns
        # Проверяем количество строк: оригинальные + сгенерированные
        expected_rows = len(crime_model.texts) + sum(len(v) for v in crime_model.generated_texts.values())
        assert len(df) == expected_rows

@patch('models.crime_model.AutoModel')
@patch('models.crime_model.AutoTokenizer')
def test_get_embeddings(mock_tokenizer, mock_model, crime_model):
    """Test _get_embeddings method"""
    # Mock model and tokenizer
    mock_model_instance = Mock()
    mock_model_instance.device = torch.device('cpu')  # Add device attribute
    mock_model.from_pretrained.return_value = mock_model_instance
    
    mock_tokenizer_instance = Mock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    
    # Test with single text
    single_text = "Тестовый текст"
    mock_tokenizer_instance.encode_plus.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_model_instance.return_value.last_hidden_state = torch.randn(1, 3, 768)
    
    embeddings = crime_model._get_embeddings(single_text, model=mock_model_instance, tokenizer=mock_tokenizer_instance)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 768  # BERT embedding size
    
    # Test with list of texts
    texts = ["Текст 1", "Текст 2", "Текст 3"]
    mock_model_instance.return_value.last_hidden_state = torch.randn(3, 3, 768)
    
    # Mock DataLoader
    with patch('torch.utils.data.DataLoader') as mock_dataloader:
        mock_dataloader.return_value = [
            {
                'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            }
        ]
        embeddings = crime_model._get_embeddings(texts, model=mock_model_instance, tokenizer=mock_tokenizer_instance)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 3  # Number of texts
        assert embeddings.shape[1] == 768  # BERT embedding size

def test_cluster_texts_with_optics(crime_model):
    """Test _cluster_texts_with_optics method"""
    # Create sample embeddings
    embeddings = np.random.rand(10, 768).astype(np.float32)
    
    # Test clustering with default parameters
    labels, clusters = crime_model._cluster_texts_with_optics(embeddings)
    assert isinstance(labels, np.ndarray)
    assert isinstance(clusters, list)
    assert len(labels) == len(embeddings)
    
    # Test clustering with custom parameters
    labels, clusters = crime_model._cluster_texts_with_optics(
        embeddings,
        min_samples=3,
        xi=0.05,
        min_cluster_size=4
    )
    assert isinstance(labels, np.ndarray)
    assert isinstance(clusters, list)
    
    # Test with torch tensor input
    torch_embeddings = torch.tensor(embeddings)
    labels, clusters = crime_model._cluster_texts_with_optics(torch_embeddings)
    assert isinstance(labels, np.ndarray)
    assert isinstance(clusters, list)

def test_get_formatted_text(crime_model):
    """Test get_formatted_text method"""
    # Test with text containing entities and weapons
    text = "Иван Иванов совершил ограбление в Москве с использованием пистолета и ножа."
    formatted = crime_model.get_formatted_text(text)
    
    # Check that formatted text contains original text
    assert text in formatted
    
    # Check that metadata is added
    assert "Оружие:" in formatted
    assert "Места:" in formatted
    assert "Участники:" in formatted
    
    # Test with empty text
    assert crime_model.get_formatted_text("") == ""
    
    # Test with None
    assert crime_model.get_formatted_text(None) == ""
    
    # Test with text without entities or weapons
    text = "Произошло обычное происшествие."
    formatted = crime_model.get_formatted_text(text)
    assert text == formatted  # Should return original text without metadata
    
    # Test with text containing negative weapon phrases
    text = "Произошло ограбление без применения оружия в Москве."
    formatted = crime_model.get_formatted_text(text)
    assert "Оружие:" not in formatted  # Should not include weapons section
    assert "Места:" in formatted  # Should still include locations

def test_crime_dataset_getitem(sample_texts_df):
    """Test CrimeDataset __getitem__ method"""
    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.encode_plus.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    dataset = CrimeDataset(sample_texts_df, mock_tokenizer)
    
    # Test getting first item
    item = dataset[0]
    assert isinstance(item, dict)
    assert 'text' in item
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['attention_mask'], torch.Tensor)
    
    # Test tokenizer was called with correct parameters
    mock_tokenizer.encode_plus.assert_called_with(
        sample_texts_df['text'].iloc[0],
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Test with custom max_len
    dataset = CrimeDataset(sample_texts_df, mock_tokenizer, max_len=64)
    item = dataset[0]
    mock_tokenizer.encode_plus.assert_called_with(
        sample_texts_df['text'].iloc[0],
        add_special_tokens=True,
        max_length=64,  # Should use custom max_len
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

def test_extract_entities_comprehensive(crime_model):
    """Test _extract_entities method with various entity types and edge cases"""
    # Test with all entity types
    text = "Иван Иванов из ООО 'Рога и Копыта' совершил ограбление в Москве с пистолетом."
    entities = crime_model._extract_entities(text)

    # Check that all expected entity types are present
    assert 'PER' in entities
    assert 'LOC' in entities
    assert 'ORG' in entities

    # Check that entities are found using any() to handle variations in extraction
    assert any('Иван' in per for per in entities['PER'])
    assert any('Москв' in loc for loc in entities['LOC'])
    assert any('Рога' in org for org in entities['ORG'])

    # Test with empty text
    entities = crime_model._extract_entities("")
    assert isinstance(entities, dict)
    assert len(entities) == 0

    # Test with text containing no entities
    entities = crime_model._extract_entities("Это обычный текст без именованных сущностей.")
    assert isinstance(entities, dict)
    assert len(entities) == 0

    # Test with text containing multiple entities of the same type
    text = "Иван Иванов и Петр Петров работают в ООО 'Рога и Копыта' и ООО 'Копыта и Рога'."
    entities = crime_model._extract_entities(text)
    assert 'PER' in entities
    assert 'ORG' in entities
    assert len(entities['PER']) >= 1  # At least one person should be found
    assert len(entities['ORG']) >= 1  # At least one organization should be found
    assert any('Иван' in per for per in entities['PER'])
    assert any('Петр' in per for per in entities['PER'])
    assert any('Рога' in org for org in entities['ORG'])

def test_cluster_texts_with_optics_parameters(crime_model):
    """Test _cluster_texts_with_optics with different parameters"""
    # Create sample embeddings
    embeddings = np.random.rand(20, 768).astype(np.float32)
    
    # Test with default parameters
    labels1, clusters1 = crime_model._cluster_texts_with_optics(embeddings)
    assert isinstance(labels1, np.ndarray)
    assert isinstance(clusters1, list)
    
    # Test with custom parameters
    labels2, clusters2 = crime_model._cluster_texts_with_optics(
        embeddings,
        min_samples=3,
        xi=0.05,
        min_cluster_size=4
    )
    assert isinstance(labels2, np.ndarray)
    assert isinstance(clusters2, list)
    
    # Test with very strict parameters (should result in fewer clusters)
    labels3, clusters3 = crime_model._cluster_texts_with_optics(
        embeddings,
        min_samples=5,
        xi=0.1,
        min_cluster_size=6
    )
    assert len(clusters3) <= len(clusters1)
    
    # Test with very loose parameters (should result in more clusters)
    labels4, clusters4 = crime_model._cluster_texts_with_optics(
        embeddings,
        min_samples=2,
        xi=0.01,
        min_cluster_size=2
    )
    assert len(clusters4) >= len(clusters1)

def test_export_to_excel_comprehensive(crime_model, tmp_path):
    """Test export_to_excel with various scenarios"""
    # Test with empty data first
    crime_model.texts = []
    crime_model.cluster_labels = []
    crime_model.clusters = []
    crime_model.generated_texts = {}
    
    with pytest.raises(ValueError, match="No data to export"):
        crime_model.export_to_excel(str(tmp_path / "empty.xlsx"))

    # Setup test data with different types of texts
    crime_model.texts = [
        "Иван Иванов совершил ограбление в Москве с пистолетом.",
        "В Санкт-Петербурге произошла кража с применением ножа.",
        "В Казани совершено нападение без оружия."
    ]
    crime_model.cluster_labels = [0, 0, 1]
    crime_model.clusters = [(0, 2), (1, 1)]
    crime_model.generated_texts = {
        0: ["Сгенерированный текст 1", "Сгенерированный текст 2"],
        1: ["Сгенерированный текст 3"]
    }

    # Test normal export
    file_path = tmp_path / "export_test1.xlsx"
    crime_model.export_to_excel(str(file_path))
    assert file_path.exists()

    # Test export with empty generated texts
    crime_model.generated_texts = {}
    file_path = tmp_path / "export_test2.xlsx"
    crime_model.export_to_excel(str(file_path))
    assert file_path.exists()

    # Test export with invalid file path
    with pytest.raises(Exception):
        crime_model.export_to_excel("/invalid/path/test.xlsx")

def test_get_embeddings_edge_cases(crime_model):
    """Test _get_embeddings with edge cases"""
    # Test with empty text - should work as BERT can handle empty strings
    embeddings = crime_model._get_embeddings("")
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 768  # BERT embedding size

    # Test with None - should convert to empty string
    embeddings = crime_model._get_embeddings(None)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 768

    # Test with empty list
    embeddings = crime_model._get_embeddings([])
    assert isinstance(embeddings, torch.Tensor)
    assert len(embeddings) == 0

    # Test with very long text
    long_text = "тест " * 1000  # Create a very long text
    embeddings = crime_model._get_embeddings(long_text)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 768

    # Test with special characters
    special_text = "!@#$%^&*()_+{}|:<>?[]\\;',./~`"
    embeddings = crime_model._get_embeddings(special_text)
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[1] == 768