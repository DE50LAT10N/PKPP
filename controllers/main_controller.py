from PyQt6.QtCore import QThread, pyqtSignal
import logging
import os
import requests
import re
from dotenv import load_dotenv
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from datetime import datetime
from pathlib import Path

from models.crime_model import CrimeModel
from utils.decorators import log_execution_time

logger = logging.getLogger('CrimeAI')

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

class DataProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.model = CrimeModel()

    @log_execution_time
    def run(self):
        try:
            logger.info(f"Starting data processing for file: {self.file_path}")
            self.progress.emit(10)
            
            result = self.model.load_data(self.file_path)
            
            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            error_msg = f"Error in data processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error.emit(error_msg)

class MainController:
    def __init__(self, view):
        self.view = view
        self.model = CrimeModel()
        self.current_cluster = None
        
        self.view.browse_button.clicked.connect(self.browse_file)
        self.view.process_button.clicked.connect(self.process_data)
        self.view.cluster_list.itemClicked.connect(self.on_cluster_selected)
        self.view.generate_button.clicked.connect(self.on_generate_clicked)
        self.view.export_button.clicked.connect(self.export_to_excel)
    
    def browse_file(self):
        file_path = self.view.get_file_path()
        if file_path:
            self.view.file_path_label.setText(file_path)
    
    def process_data(self):
        file_path = self.view.file_path_label.text()
        if not file_path or file_path == "No file selected":
            self.view.show_error("Please select a file first")
            return
            
        self.view.progress_bar.setValue(0)
        self.processor = DataProcessor(file_path)
        self.processor.progress.connect(self.view.update_progress)
        self.processor.finished.connect(self.on_data_processed)
        self.processor.error.connect(self.view.show_error)
        self.processor.start()
    
    def on_data_processed(self, result):
        try:
            self.model.texts = result['texts']
            self.model.embeddings = result['embeddings']
            self.model.cluster_labels = result['cluster_labels']
            self.model.clusters = result['clusters']
            
            self.view.update_cluster_list(self.model.clusters)
            self.view.generate_button.setEnabled(True)
            logger.info("Data processing completed successfully")
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.view.show_error(error_msg)
    
    def on_cluster_selected(self, item):
        if not item:
            return
        
        try:
            cluster_text = item.text()
            cluster_label = int(cluster_text.split()[1])
            self.current_cluster = cluster_label
            
            cluster_texts = self.model.get_cluster_texts(cluster_label)
            self.view.update_original_texts(cluster_texts)
            
            generated_texts = self.model.get_generated_texts(cluster_label)
            if generated_texts:
                self.view.update_generated_texts(generated_texts)
            logger.info(f"Cluster {cluster_label} selected successfully")
        except Exception as e:
            error_msg = f"Error selecting cluster: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.view.show_error(error_msg)
    
    def on_generate_clicked(self):
        if self.current_cluster is None:
            error_msg = "Please select a cluster first"
            logger.warning(error_msg)
            self.view.show_error(error_msg)
            return
        
        try:
            cluster_texts = self.model.get_cluster_texts(self.current_cluster)
            
            self.view.generate_button.setEnabled(False)
            
            generated_text = self.predict_next_crime(cluster_texts)
            
            for text in generated_text:
                self.model.add_generated_text(self.current_cluster, text)
            
            self.view.update_generated_texts([generated_text])
            logger.info(f"Prediction generated successfully for cluster {self.current_cluster}")
            
        except Exception as e:
            error_msg = f"Error generating prediction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.view.show_error(error_msg)
        finally:
            self.view.generate_button.setEnabled(True)

    
    def export_to_excel(self):
        if not self.model.texts:
            error_msg = "No data to export"
            logger.warning(error_msg)
            self.view.show_error(error_msg)
            return
            
        file_path = self.view.get_save_path()
        if file_path:
            try:
                self.model.export_to_excel(file_path)
                success_msg = "Data exported successfully"
                logger.info(success_msg)
                self.view.show_success(success_msg)
            except Exception as e:
                error_msg = f"Error exporting data: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.view.show_error(error_msg)
    
    @log_execution_time
    def predict_next_crime(self, cluster_texts: list) -> str:
        if not cluster_texts or not isinstance(cluster_texts, (list, tuple)):
            logger.warning("Invalid cluster texts provided")
            return ""
        
        try:
            # Структура для сбора информации о преступлениях
            crime_info = {
                'entities': {
                    'LOC': set(),    # Места преступлений
                    'PER': set(),    # Участники
                    'PRODUCT': set(), # Оружие/предметы
                    'ORG': set(),    # Организации
                    'MONEY': set(),  # Денежные суммы
                    'DATE': set()    # Даты
                },
                'crime_details': {
                    'locations': {},      # Частота мест
                    'participants': [],   # Количество участников
                    'damages': [],        # Суммы ущерба
                    'times': set(),       # Времена преступлений
                    'weapons': set()      # Использованное оружие
                }
            }
            
            # Анализ каждого текста в кластере
            for text in cluster_texts:
                # Лингвистический анализ через Natasha
                doc = Doc(text)
                doc.segment(segmenter)
                doc.tag_morph(morph_tagger)
                doc.parse_syntax(syntax_parser)
                doc.tag_ner(ner_tagger)
                
                # Сбор именованных сущностей
                for span in doc.spans:
                    if span.type in crime_info['entities']:
                        crime_info['entities'][span.type].add(span.text)
                        # Дополнительная обработка для мест и оружия
                        if span.type == 'LOC':
                            crime_info['crime_details']['locations'][span.text] = \
                                crime_info['crime_details']['locations'].get(span.text, 0) + 1
                        elif span.type == 'PRODUCT':
                            crime_info['crime_details']['weapons'].add(span.text)
                
                # Подсчет участников
                participants = [span for span in doc.spans if span.type == 'PER']
                crime_info['crime_details']['participants'].append(len(participants))
                
                # Извлечение денежных сумм
                money_patterns = re.findall(r'\d+(?:\s\d{3})*(?:\.\d+)?\s*(?:руб|₽|тыс|млн|млрд)', text)
                for amount in money_patterns:
                    try:
                        clean_amount = float(re.sub(r'[^\d.]', '', amount))
                        crime_info['crime_details']['damages'].append(clean_amount)
                    except (ValueError, TypeError):
                        continue
                
                # Извлечение времени
                time_patterns = re.findall(r'\d{1,2}:\d{2}|\d{1,2} час|\d{1,2}:\d{2} час', text.lower())
                crime_info['crime_details']['times'].update(time_patterns)
            
            # Анализ собранных данных
            most_common_location = max(crime_info['crime_details']['locations'].items(), 
                                     key=lambda x: x[1], default=(None, 0))[0]
            avg_participants = sum(crime_info['crime_details']['participants']) / len(crime_info['crime_details']['participants']) \
                if crime_info['crime_details']['participants'] else 1
            avg_damage = sum(crime_info['crime_details']['damages']) / len(crime_info['crime_details']['damages']) \
                if crime_info['crime_details']['damages'] else 0
            
            # Формирование промпта для GPT с примерами текстов
            prompt = f"""Ты - опытный криминолог, анализирующий серию преступлений. На основе анализа предыдущих преступлений, 
составь прогноз следующего преступления в серии. Твой прогноз должен быть написан в том же стиле, что и примеры ниже.

Примеры преступлений из этой серии:
{chr(10).join(f'Пример {i+1}:{chr(10)}{text}{chr(10)}' for i, text in enumerate(cluster_texts[:3]))}

Анализ предыдущих преступлений:
1. Места преступлений: {', '.join(crime_info['entities']['LOC'])}
2. Участники: {', '.join(crime_info['entities']['PER'])}
3. Использованное оружие: {', '.join(crime_info['crime_details']['weapons'])}
4. Организации: {', '.join(crime_info['entities']['ORG'])}
5. Времена преступлений: {', '.join(crime_info['crime_details']['times'])}
6. Наиболее частое место: {most_common_location or 'неизвестно'}
7. Среднее количество участников: {avg_participants:.1f}
8. Средний ущерб: {avg_damage:.2f} руб.

Требования к предсказанию:
1. Напиши прогноз в том же стиле и формате, что и примеры выше
2. Используй похожие формулировки и структуру предложений
3. Сохрани характерные особенности описания преступлений из примеров
4. Укажи вероятное время следующего преступления
5. Опиши вероятное место совершения
6. Предположи количество участников
7. Оцени возможный ущерб
8. Укажи вероятное оружие/способ совершения
9. Обоснуй предсказание на основе анализа серии
10. Если какие-то параметры неизвестны, укажи это

Прогноз следующего преступления в серии:"""
            
            # Сохранение промпта в файл
            prompts_dir = Path("prompts")
            prompts_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_filename = prompts_dir / f"prompt_{timestamp}.txt"
            
            try:
                with open(prompt_filename, "w", encoding="utf-8") as f:
                    f.write(prompt)
                logger.info(f"Prompt saved to {prompt_filename}")
            except Exception as e:
                logger.error(f"Error saving prompt to file: {str(e)}", exc_info=True)
            
            logger.info("Generating crime prediction using GPT-4")
            prediction = self._get_gpt4_response(prompt)
            logger.info("Received prediction from GPT-4")
            
            return prediction if prediction else ""
            
        except Exception as e:
            logger.error(f"Error generating crime prediction: {str(e)}", exc_info=True)
            return ""
    
    def _get_gpt4_response(self, prompt):
        try:
            load_dotenv()
            api_key = os.getenv('API_KEY')
            if not api_key:
                error_msg = "GPT4_API_KEY not found in environment variables"
                logger.error(error_msg)
                self.view.show_error(error_msg)
                return ""

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            logger.info("Sending request to GPT-4 API")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                self.view.show_error(error_msg)
                return ""
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                error_msg = "Invalid response format from API"
                logger.error(error_msg)
                self.view.show_error(error_msg)
                return ""
                
            logger.info("Successfully received response from GPT-4 API")
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            error_msg = f"Error getting GPT-4 response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.view.show_error(error_msg)
            return ""
    