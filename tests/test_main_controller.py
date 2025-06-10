import pytest
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QListWidgetItem
from controllers.main_controller import MainController, DataProcessor
import logging

@pytest.fixture
def mock_view():
    view = Mock()
    view.browse_button = Mock()
    view.process_button = Mock()
    view.cluster_list = Mock()
    view.generate_button = Mock()
    view.export_button = Mock()
    view.file_path_label = Mock()
    view.progress_bar = Mock()
    view.loading_widget = Mock()
    view.show_error = Mock()
    view.update_progress = Mock()
    view.update_cluster_list = Mock()
    view.update_original_texts = Mock()
    view.update_generated_texts = Mock()
    view.show_success = Mock()
    view.get_file_path = Mock()
    view.get_save_path = Mock()
    return view

@pytest.fixture
def controller(mock_view):
    return MainController(mock_view)

def test_browse_file(controller, mock_view):
    # Arrange
    expected_path = "test/path/file.txt"
    mock_view.get_file_path.return_value = expected_path
    
    # Act
    controller.browse_file()
    
    # Assert
    mock_view.get_file_path.assert_called_once()
    mock_view.file_path_label.setText.assert_called_once_with(expected_path)

def test_process_data_no_file(controller, mock_view):
    # Arrange
    mock_view.file_path_label.text.return_value = "No file selected"
    
    # Act
    controller.process_data()
    
    # Assert
    mock_view.show_error.assert_called_once_with("Please select a file first")
    assert not hasattr(controller, 'processor')

def test_process_data_with_file(controller, mock_view):
    # Arrange
    file_path = "test/path/file.txt"
    mock_view.file_path_label.text.return_value = file_path
    
    # Act
    controller.process_data()
    
    # Assert
    assert isinstance(controller.processor, DataProcessor)
    assert controller.processor.file_path == file_path
    mock_view.progress_bar.setValue.assert_called_once_with(0)

def test_on_data_processed(controller, mock_view):
    # Arrange
    result = {
        'texts': ['text1', 'text2'],
        'embeddings': [[1, 2], [3, 4]],
        'cluster_labels': [0, 1],
        'clusters': {0: ['text1'], 1: ['text2']}
    }
    
    # Act
    controller.on_data_processed(result)
    
    # Assert
    assert controller.model.texts == result['texts']
    assert controller.model.embeddings == result['embeddings']
    assert controller.model.cluster_labels == result['cluster_labels']
    assert controller.model.clusters == result['clusters']
    mock_view.update_cluster_list.assert_called_once_with(result['clusters'])
    mock_view.generate_button.setEnabled.assert_called_once_with(True)

def test_on_cluster_selected(controller, mock_view):
    # Arrange
    item = QListWidgetItem("Cluster 1")
    cluster_texts = ['text1', 'text2']
    generated_texts = ['generated1']
    controller.model = Mock()
    controller.model.get_cluster_texts.return_value = cluster_texts
    controller.model.get_generated_texts.return_value = generated_texts
    
    # Act
    controller.on_cluster_selected(item)
    
    # Assert
    assert controller.current_cluster == 1
    controller.model.get_cluster_texts.assert_called_once_with(1)
    mock_view.update_original_texts.assert_called_once_with(cluster_texts)
    mock_view.update_generated_texts.assert_called_once_with(generated_texts)


def test_on_generate_clicked_no_cluster(controller, mock_view):
    # Arrange
    controller.current_cluster = None
    controller.model = Mock()
    controller.model.get_cluster_texts = Mock(return_value=[])
    
    # Act
    controller.on_generate_clicked()
    
    # Assert
    mock_view.generate_button.setEnabled.assert_not_called()
    mock_view.loading_widget.start.assert_not_called()
    mock_view.loading_widget.stop.assert_not_called()
    mock_view.update_generated_texts.assert_not_called()


@patch('controllers.main_controller.MainController.predict_next_crime')
def test_on_generate_clicked_with_cluster(mock_predict, controller, mock_view):
    # Arrange
    controller.current_cluster = 1
    cluster_texts = ['text1', 'text2']
    generated_texts = ["generated prediction 1", "generated prediction 2"]
    controller.model = Mock()
    controller.model.get_cluster_texts.return_value = cluster_texts
    mock_predict.return_value = generated_texts

    # Reset mocks before test
    mock_view.generate_button.setEnabled.reset_mock()
    mock_view.loading_widget.start.reset_mock()
    mock_view.loading_widget.stop.reset_mock()
    mock_view.update_generated_texts.reset_mock()

    # Act
    controller.on_generate_clicked()

    # Assert
    assert mock_view.generate_button.setEnabled.call_count == 2  # False then True
    mock_view.loading_widget.start.assert_called_once()
    mock_predict.assert_called_once_with(cluster_texts)
    assert controller.model.add_generated_text.call_count == len(generated_texts)
    for text in generated_texts:
        controller.model.add_generated_text.assert_any_call(1, text)
    mock_view.update_generated_texts.assert_called_once_with([generated_texts])
    mock_view.loading_widget.stop.assert_called_once()

def test_export_to_excel_no_data(controller, mock_view):
    # Arrange
    controller.model = Mock()
    controller.model.texts = []
    
    # Act
    controller.export_to_excel()
    
    # Assert
    mock_view.show_error.assert_called_once_with("No data to export")

def test_export_to_excel_success(controller, mock_view):
    # Arrange
    controller.model = Mock()
    controller.model.texts = ['text1', 'text2']
    mock_view.get_save_path.return_value = "test/path/export.xlsx"
    
    # Act
    controller.export_to_excel()
    
    # Assert
    controller.model.export_to_excel.assert_called_once_with("test/path/export.xlsx")
    mock_view.show_success.assert_called_once_with("Data exported successfully")


@patch('controllers.main_controller.MainController._get_gpt4_response')
def test_predict_next_crime(mock_gpt4_response, controller):
    # Arrange
    cluster_texts = ['text1', 'text2']
    expected_response = "Generated prediction text"
    mock_gpt4_response.return_value = expected_response
    
    # Act
    result = controller.predict_next_crime(cluster_texts)
    
    # Assert
    assert result == expected_response
    mock_gpt4_response.assert_called_once()

def test_predict_next_crime_empty_input(controller):
    # Act
    result = controller.predict_next_crime([])
    
    # Assert
    assert result == ""

@patch('controllers.main_controller.MainController._get_gpt4_response')
def test_predict_next_crime_error_handling(mock_gpt4_response, controller, caplog):
    """Test error handling in predict_next_crime"""
    # Clear any existing logs
    caplog.clear()
    
    # Test with API error
    mock_gpt4_response.side_effect = Exception("API Error")
    cluster_texts = ['text1', 'text2']
    
    with caplog.at_level(logging.ERROR):
        result = controller.predict_next_crime(cluster_texts)
        assert "Error generating crime prediction: API Error" in caplog.text
        assert result == ""  # Should return empty string on error
        caplog.clear()

    # Test with empty input
    with caplog.at_level(logging.WARNING):
        result = controller.predict_next_crime([])
        assert result == ""
        assert "Invalid cluster texts provided" in caplog.text
        caplog.clear()

    # Test with None input
    with caplog.at_level(logging.WARNING):
        result = controller.predict_next_crime(None)
        assert result == ""
        assert "Invalid cluster texts provided" in caplog.text
        caplog.clear()

    # Test with invalid input type
    with caplog.at_level(logging.WARNING):
        result = controller.predict_next_crime("not a list")
        assert result == ""
        assert "Invalid cluster texts provided" in caplog.text
        caplog.clear()

    # Verify that no logs remain
    assert not caplog.records

@patch('controllers.main_controller.MainController._get_gpt4_response')
def test_predict_next_crime_pattern_analysis(mock_gpt4_response, controller):
    """Test pattern analysis in predict_next_crime"""
    # Test with texts containing various patterns
    cluster_texts = [
        "Иван Иванов совершил ограбление в Москве с пистолетом 15 января.",
        "Петр Петров совершил разбой в Санкт-Петербурге с ножом 16 января.",
        "Сидор Сидоров совершил кражу в Казани без оружия 17 января."
    ]

    # Mock GPT-4 response
    mock_gpt4_response.return_value = "Сгенерированный текст с паттернами"

    # Generate prediction and verify analysis
    result = controller.predict_next_crime(cluster_texts)
    
    # Assert
    assert result == "Сгенерированный текст с паттернами"
    mock_gpt4_response.assert_called_once()
    
    # Verify that the prompt contains analyzed patterns
    prompt = mock_gpt4_response.call_args[0][0]
    assert "Москве" in prompt
    assert "Санкт-Петербурге" in prompt
    assert "Казани" in prompt
    assert "Иван Иванов" in prompt
    assert "Петр Петров" in prompt
    assert "Сидор Сидоров" in prompt
    assert "пистолетом" in prompt
    assert "ножом" in prompt

@patch('requests.post')
def test_get_gpt4_response(mock_post, controller):
    """Test _get_gpt4_response method"""
    # Test successful API call
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "output": [{
            "content": [{
                "text": "Generated text"
            }]
        }]
    }
    mock_post.return_value = mock_response

    result = controller._get_gpt4_response("Test prompt")
    assert result == "Generated text"

    # Test API error
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    with pytest.raises(Exception) as exc_info:
        controller._get_gpt4_response("Test prompt")
    assert "Request failed with status 400" in str(exc_info.value)

    # Test invalid response format
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid": "format"}
    with pytest.raises(Exception) as exc_info:
        controller._get_gpt4_response("Test prompt")
    assert "Ответ не содержит ключ 'output'" in str(exc_info.value)

def test_data_processor_error_handling():
    """Test DataProcessor error handling"""
    processor = DataProcessor("nonexistent_file.xlsx")
    
    # Mock the view to capture signals
    mock_view = Mock()
    processor.error.connect(mock_view.show_error)
    processor.finished.connect(mock_view.on_finished)
    
    # Run processor
    processor.run()
    
    # Verify error signal was emitted
    mock_view.show_error.assert_called_once()
    assert "Error in data processing" in mock_view.show_error.call_args[0][0]
    mock_view.on_finished.assert_not_called()

@patch('models.crime_model.CrimeModel.load_data')
def test_data_processor_success(mock_load_data):
    """Test DataProcessor successful execution"""
    # Mock successful data loading
    mock_load_data.return_value = {
        'texts': ['text1', 'text2'],
        'embeddings': [[1, 2], [3, 4]],
        'cluster_labels': [0, 1],
        'clusters': {0: ['text1'], 1: ['text2']}
    }
    
    processor = DataProcessor("test_file.xlsx")
    
    # Mock the view to capture signals
    mock_view = Mock()
    processor.progress.connect(mock_view.update_progress)
    processor.finished.connect(mock_view.on_finished)
    processor.error.connect(mock_view.show_error)
    
    # Run processor
    processor.run()
    
    # Verify signals
    assert mock_view.update_progress.call_count >= 2  # At least start and end progress
    mock_view.on_finished.assert_called_once()
    mock_view.show_error.assert_not_called() 