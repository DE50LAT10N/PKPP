from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                           QFileDialog, QMessageBox, QTabWidget, QListWidget,
                           QProgressBar, QSplitter, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crime Analysis System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QTabWidget::pane {
                border: 1px solid #dcdde1;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #dcdde1;
                color: #2c3e50;
                padding: 8px 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QTextEdit {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QListWidget {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                background-color: white;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #dcdde1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QProgressBar {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        upload_tab = QWidget()
        upload_layout = QVBoxLayout(upload_tab)
        upload_layout.setSpacing(15)
        
        file_frame = QFrame()
        file_frame.setFrameShape(QFrame.Shape.StyledPanel)
        file_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        file_layout = QHBoxLayout(file_frame)
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
        """)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setIcon(QIcon("icons/folder.png"))
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.browse_button)
        upload_layout.addWidget(file_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                height: 25px;
                font-size: 14px;
            }
        """)
        upload_layout.addWidget(self.progress_bar)
        
        self.process_button = QPushButton("Process Data")
        self.process_button.setIcon(QIcon("icons/process.png"))
        upload_layout.addWidget(self.process_button)
        
        tabs.addTab(upload_tab, "Data Upload")
        
        # Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setSpacing(15)
        
        cluster_frame = QFrame()
        cluster_frame.setFrameShape(QFrame.Shape.StyledPanel)
        cluster_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        cluster_layout = QVBoxLayout(cluster_frame)
        
        cluster_label = QLabel("Clusters")
        cluster_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        cluster_layout.addWidget(cluster_label)
        
        self.cluster_list = QListWidget()
        cluster_layout.addWidget(self.cluster_list)
        
        analysis_layout.addWidget(cluster_frame)
        
        details_frame = QFrame()
        details_frame.setFrameShape(QFrame.Shape.StyledPanel)
        details_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 15px;
            }
        """)
        details_layout = QVBoxLayout(details_frame)
        
        details_label = QLabel("Cluster Details")
        details_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        details_layout.addWidget(details_label)
        
        details_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        original_frame = QFrame()
        original_frame.setFrameShape(QFrame.Shape.StyledPanel)
        original_layout = QVBoxLayout(original_frame)
        
        original_label = QLabel("Original Texts")
        original_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        original_layout.addWidget(original_label)
        
        self.original_texts = QTextEdit()
        self.original_texts.setReadOnly(True)
        self.original_texts.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        original_layout.addWidget(self.original_texts)
        
        details_splitter.addWidget(original_frame)
        
        generated_frame = QFrame()
        generated_frame.setFrameShape(QFrame.Shape.StyledPanel)
        generated_layout = QVBoxLayout(generated_frame)
        
        generated_label = QLabel("Generated Texts")
        generated_label.setStyleSheet("""
            QLabel {
                color: #2c3e50;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        generated_layout.addWidget(generated_label)
        
        self.generated_texts_widget = QTextEdit()
        self.generated_texts_widget.setReadOnly(True)
        self.generated_texts_widget.setStyleSheet("""
            QTextEdit {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        generated_layout.addWidget(self.generated_texts_widget)
        
        details_splitter.addWidget(generated_frame)
        
        details_layout.addWidget(details_splitter)
        analysis_layout.addWidget(details_frame)
        
        buttons_frame = QFrame()
        buttons_frame.setFrameShape(QFrame.Shape.StyledPanel)
        buttons_layout = QHBoxLayout(buttons_frame)
        
        self.generate_button = QPushButton("Predict crime")
        self.generate_button.setIcon(QIcon("icons/predict.png"))
        self.generate_button.setEnabled(False)
        buttons_layout.addWidget(self.generate_button)
        
        self.export_button = QPushButton("Export to Excel")
        self.export_button.setIcon(QIcon("icons/export.png"))
        buttons_layout.addWidget(self.export_button)
        
        analysis_layout.addWidget(buttons_frame)
        
        tabs.addTab(analysis_tab, "Analysis")
        
    def update_cluster_list(self, clusters):
        self.cluster_list.clear()
        for cluster_label, cluster_size in clusters:
            self.cluster_list.addItem(f"Кластер {cluster_label} (Размер: {cluster_size})")
    
    def update_original_texts(self, texts):
        self.original_texts.clear()
        for text in texts:
            self.original_texts.append(text)
            self.original_texts.append("─" * 50)
    
    def update_generated_texts(self, texts):
        self.generated_texts_widget.clear()
        for text in texts:
            self.generated_texts_widget.append(text)
            self.generated_texts_widget.append("─" * 50)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
    
    def show_success(self, message):
        QMessageBox.information(self, "Success", message)
    
    def get_file_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            "",
            "Excel Files (*.xlsx *.xls)"
        )
        return file_path
    
    def get_save_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Excel File",
            "",
            "Excel Files (*.xlsx)"
        )
        return file_path 