# main.py
import sys
import os
import cv2
import numpy as np
import json
import re
import sqlite3
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from collections import defaultdict
from fpdf import FPDF
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                            QFileDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox,
                            QProgressBar, QTextEdit, QFrame, QSizePolicy, QDialog, QComboBox,
                            QAction, QToolBar, QTabWidget, QMenu, QInputDialog, QFormLayout,
                            QDialogButtonBox, QSystemTrayIcon, QStyle)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPalette, QKeySequence
from PyQt5.QtNetwork import QNetworkRequest, QNetworkAccessManager, QNetworkReply
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==================== 云服务配置管理 ====================
class CloudConfigManager:
    def __init__(self):
        self.config_path = resource_path('cloud_config.json')
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            "enable": False,
            "endpoint": "https://api.example.com",
            "api_key": "",
            "sync_interval": 3600
        }

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f)

# ==================== 动态地图标记 ====================
class MapBridge(QObject):
    add_marker = pyqtSignal(float, float, str, str)

class MapWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = MapBridge()
        self.page().profile().clearHttpCache()
        self.load_map_template()

    def load_map_template(self):
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>#map { height: 100%; }</style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map = L.map('map').setView([30.5928, 114.3055], 6);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
                var markers = [];
                
                function addMarker(lat, lng, title, desc) {
                    var marker = L.marker([lat, lng]).addTo(map)
                        .bindPopup(`<b>${title}</b><br>${desc}`);
                    markers.push(marker);
                }
            </script>
        </body>
        </html>
        """
        self.setHtml(html)
        self.bridge.add_marker.connect(self.add_marker)

    def add_marker(self, lat, lng, title, desc):
        js = f"addMarker({lat}, {lng}, '{title}', '{desc}');"
        self.page().runJavaScript(js)

# ==================== 云端同步服务 ====================
class CloudSync(QThread):
    sync_progress = pyqtSignal(int)
    sync_finished = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.manager = QNetworkAccessManager()

    def run(self):
        if not self.config['enable']:
            return

        # 同步模型
        model_file = open(resource_path('disaster_model.h5'), 'rb')
        request = QNetworkRequest(QUrl(f"{self.config['endpoint']}/upload"))
        request.setRawHeader(b"Authorization", self.config['api_key'].encode())
        reply = self.manager.put(request, model_file.read())
        reply.finished.connect(lambda: self.handle_reply(reply))

    def handle_reply(self, reply):
        if reply.error() == QNetworkReply.NoError:
            self.sync_finished.emit(True)
        else:
            self.sync_finished.emit(False)

# ==================== 功能栏实现 ====================
class RibbonToolBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self.setStyleSheet("""
            QToolBar { 
                background: #F3F3F3;
                border-bottom: 1px solid #DCDCDC;
                spacing: 4px;
                padding: 4px;
            }
            QToolButton { 
                padding: 8px;
                min-width: 80px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: #E5F1FB;
            }
        """)

class FileGroup(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.new_btn = QPushButton("新建")
        self.open_btn = QPushButton("打开")
        self.save_btn = QPushButton("保存")
        layout.addWidget(self.new_btn)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

# ==================== 主界面 ====================
class DisasterAssessmentSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cloud_config = CloudConfigManager().config
        self.init_ui()
        self.init_model()
        self.init_cloud()

    def init_ui(self):
        self.setWindowTitle("农业灾害智能评估系统 v5.0")
        self.setGeometry(100, 100, 1280, 800)
        
        # 功能区设置
        self.create_ribbon()
        
        # 主工作区
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # 左侧面板
        left_panel = QFrame()
        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.btn_upload = QPushButton("上传照片")
        self.map_widget = MapWidget()
        tab_widget = QTabWidget()
        tab_widget.addTab(self.image_label, "图片预览")
        tab_widget.addTab(self.map_widget, "灾害地图")
        left_layout.addWidget(tab_widget)
        left_panel.setLayout(left_layout)
        
        # 右侧面板
        right_panel = QFrame()
        right_layout = QVBoxLayout()
        self.result_display = QTextEdit()
        self.policy_display = QTextEdit()
        right_layout.addWidget(QLabel("评估报告"))
        right_layout.addWidget(self.result_display)
        right_layout.addWidget(QLabel("保险政策"))
        right_layout.addWidget(self.policy_display)
        right_panel.setLayout(right_layout)
        
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_ribbon(self):
        # 文件功能区
        file_group = FileGroup(self)
        
        # 云服务功能区
        cloud_group = QWidget()
        cloud_layout = QHBoxLayout()
        self.sync_btn = QPushButton("立即同步")
        self.cloud_toggle = QPushButton("启用云服务")
        self.cloud_toggle.setCheckable(True)
        cloud_layout.addWidget(self.sync_btn)
        cloud_layout.addWidget(self.cloud_toggle)
        cloud_group.setLayout(cloud_layout)
        
        # 构建功能区
        ribbon = RibbonToolBar()
        ribbon.addWidget(file_group)
        ribbon.addSeparator()
        ribbon.addWidget(cloud_group)
        self.addToolBar(Qt.TopToolBarArea, ribbon)

    def init_model(self):
        if not os.path.exists(resource_path('disaster_model.h5')):
            self.statusBar().showMessage("正在初始化模型...")
            base_model = EfficientNetB0(weights='imagenet', include_top=False)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(3, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.save(resource_path('disaster_model.h5'))

    def init_cloud(self):
        self.cloud_sync = CloudSync(self.cloud_config)
        self.cloud_sync.sync_finished.connect(self.handle_sync_result)
        self.sync_btn.clicked.connect(self.cloud_sync.start)
        
        # 自动同步定时器
        self.sync_timer = QTimer()
        self.sync_timer.timeout.connect(self.cloud_sync.start)
        self.sync_timer.start(self.cloud_config['sync_interval'] * 1000)

    def handle_sync_result(self, success):
        if success:
            self.statusBar().showMessage("云同步成功！", 5000)
        else:
            self.statusBar().showMessage("云同步失败，请检查配置", 5000)

    # ...其余功能实现参考之前代码...

    def load_image(self):
        # 加载图片后自动添加地图标记
        file_path, _ = QFileDialog.getOpenFileName()
        if file_path:
            # 模拟获取地理坐标（实际需集成GPS信息）
            lat, lng = np.random.uniform(24.0, 40.0), np.random.uniform(110.0, 122.0)
            self.map_widget.add_marker(lat, lng, "新灾害点", "2023年水灾")

# ==================== 云配置对话框 ====================
class CloudConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("云服务配置")
        layout = QFormLayout()
        
        self.enable_check = QCheckBox("启用云同步")
        self.endpoint_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        
        layout.addRow("启用服务", self.enable_check)
        layout.addRow("API地址", self.endpoint_input)
        layout.addRow("API密钥", self.api_key_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = DisasterAssessmentSystem()
    window.show()
    sys.exit(app.exec_())