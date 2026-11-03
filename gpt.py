"""
GPT-OSS-Safeguard Chat — красивый GUI для запуска нейросети через Ollama.
Требуется: pip install ollama PyQt5
"""

import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QFrame,
    QSplitter, QMessageBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QIcon, QTextCursor, QColor, QPalette

import ollama


# ──────────────────────────────────────────────
#  Сигналы для потокобезопасного обновления UI
# ──────────────────────────────────────────────
class StreamSignals(QObject):
    chunk_received = pyqtSignal(str)
    stream_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)


# ──────────────────────────────────────────────
#  Главное окно
# ──────────────────────────────────────────────
class GPTOSSChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛡️ GPT-OSS-Safeguard Chat")
        self.setMinimumSize(900, 650)
        self.conversation_history = []
        self.is_generating = False
        self.current_model = "gpt-oss-safeguard:20b"

        self.signals = StreamSignals()
        self.signals.chunk_received.connect(self._append_chunk)
        self.signals.stream_finished.connect(self._on_stream_finished)
        self.signals.error_occurred.connect(self._on_error)

        self._setup_ui()
        self._apply_styles()
        self._check_ollama()

    # ──────────────────────────────────────────
    #  Построение интерфейса
    # ──────────────────────────────────────────
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Верхняя панель ──
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 14)

        title = QLabel("🛡️ GPT-OSS-Safeguard")
        title.setObjectName("title")

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelCombo")
        self.model_combo.addItems([
            "gpt-oss-safeguard:20b",
            "gpt-oss-safeguard:120b",
        ])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_label = QLabel("Проверка...")
        self.status_label.setObjectName("statusLabel")

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Модель:"))
        header_layout.addWidget(self.model_combo)
        header_layout.addSpacing(16)
        header_layout.addWidget(self.status_dot)
        header_layout.addWidget(self.status_label)

        # ── Область чата ──
        self.chat_display = QTextEdit()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 11))

        # ── Нижняя панель ввода ──
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(20, 12, 20, 12)

        self.input_field = QLineEdit()
        self.input_field.setObjectName("inputField")
        self.input_field.setPlaceholderText("Введите сообщение...")
        self.input_field.setFont(QFont("Segoe UI", 11))
        self.input_field.returnPressed.connect(self._send_message)

        self.send_btn = QPushButton("Отправить")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_message)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_chat)

        input_layout.addWidget(self.input_field, stretch=1)
        input_layout.addWidget(self.send_btn)
        input_layout.addWidget(self.clear_btn)

        # ── Сборка ──
        root_layout.addWidget(header)
        root_layout.addWidget(self.chat_display, stretch=1)
        root_layout.addWidget(input_frame)

        # ── Статусбар ──
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Готово")

    # ──────────────────────────────────────────
    #  Стили (QSS)
    # ──────────────────────────────────────────
    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #0f0f13;
            }
            #header {
                background: #1a1a24;
                border-bottom: 1px solid #2a2a3a;
            }
            #title {
                font-size: 20px;
                font-weight: bold;
                color: #e0e0ff;
                letter-spacing: 0.5px;
            }
            #statusDot {
                font-size: 14px;
                color: #888;
            }
            #statusLabel {
                color: #999;
                font-size: 12px;
            }
            #modelCombo {
                background: #252535;
                color: #ddd;
                border: 1px solid #3a3a50;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
                min-width: 180px;
            }
            #modelCombo::drop-down {
                border: none;
                width: 20px;
            }
            #modelCombo QAbstractItemView {
                background: #252535;
                color: #ddd;
                selection-background-color: #3a3a60;
            }
            #chatDisplay {
                background: #0f0f13;
                color: #d0d0e0;
                border: none;
                padding: 20px;
                selection-background-color: #3a3a60;
            }
            #inputFrame {
                background: #1a1a24;
                border-top: 1px solid #2a2a3a;
            }
            #inputField {
                background: #252535;
                color: #e0e0f0;
                border: 1px solid #3a3a50;
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 13px;
            }
            #inputField:focus {
                border: 1px solid #6c5ce7;
            }
            #sendBtn {
                background: #6c5ce7;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 22px;
                font-weight: bold;
                font-size: 13px;
            }
            #sendBtn:hover {
                background: #7d6ef0;
            }
            #sendBtn:pressed {
                background: #5a4bd1;
            }
            #sendBtn:disabled {
                background: #3a3a50;
                color: #666;
            }
            #clearBtn {
                background: transparent;
                color: #888;
                border: 1px solid #3a3a50;
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 13px;
            }
            #clearBtn:hover {
                background: #252535;
                color: #ccc;
            }
            QStatusBar {
                background: #1a1a24;
                color: #666;
                font-size: 11px;
            }
            QScrollBar:vertical {
                background: #0f0f13;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a50;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a4a65;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

    # ──────────────────────────────────────────
    #  Проверка подключения к Ollama
    # ──────────────────────────────────────────
    def _check_ollama(self):
        def check():
            try:
                client = ollama.Client(host="http://localhost:11434")
                models = client.list()
                model_names = [m["name"] for m in models.get("models", [])]
                self.signals.chunk_received.emit(
                    f"__STATUS__:OK|{', '.join(model_names)}"
                )
            except Exception as e:
                self.signals.chunk_received.emit(f"__STATUS__:FAIL|{e}")

        threading.Thread(target=check, daemon=True).start()

    # ──────────────────────────────────────────
    #  Обработка смены модели
    # ──────────────────────────────────────────
    def _on_model_changed(self, name):
        self.current_model = name
        self.statusBar().showMessage(f"Модель: {name}")

    # ──────────────────────────────────────────
    #  Отправка сообщения
    # ──────────────────────────────────────────
    def _send_message(self):
        text = self.input_field.text().strip()
        if not text or self.is_generating:
            return

        self.input_field.clear()
        self.is_generating = True
        self.send_btn.setEnabled(False)

        # Пользовательское сообщение
        self._append_html(
            f'<div style="margin:8px 0;">'
            f'<span style="color:#6c5ce7;font-weight:bold;">Вы</span><br>'
            f'<span style="color:#d0d0e0;">{self._escape(text)}</span>'
            f'</div>'
        )

        self.conversation_history.append({"role": "user", "content": text})

        # Запуск генерации в отдельном потоке
        threading.Thread(
            target=self._generate_response, daemon=True
        ).start()

    # ──────────────────────────────────────────
    #  Генерация ответа (поток)
    # ──────────────────────────────────────────
    def _generate_response(self):
        try:
            client = ollama.Client(host="http://localhost:11434")

            self.signals.chunk_received.emit(
                f'<div style="margin:8px 0;">'
                f'<span style="color:#00b894;font-weight:bold;">'
                f'🛡️ Safeguard</span><br>'
            )

            stream = client.chat(
                model=self.current_model,
                messages=self.conversation_history,
                stream=True,
                options={"temperature": 0.7}
            )

            full_response = ""
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_response += content
                    self.signals.chunk_received.emit(
                        self._escape(content).replace("\n", "<br>")
                    )

            self.signals.chunk_received.emit("</div>")

            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            self.signals.error_occurred.emit(str(e))
        finally:
            self.signals.stream_finished.emit()

    # ──────────────────────────────────────────
    #  Слоты
    # ──────────────────────────────────────────
    def _append_chunk(self, chunk):
        if chunk.startswith("__STATUS__:"):
            self._handle_status(chunk)
            return
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(chunk)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _handle_status(self, raw):
        parts = raw.replace("__STATUS__:", "").split("|", 1)
        ok = parts[0] == "OK"
        detail = parts[1] if len(parts) > 1 else ""

        if ok:
            self.status_dot.setStyleSheet("color:#00b894;")
            self.status_label.setText("Ollama подключён")
            self.statusBar().showMessage(f"Доступные модели: {detail}")
        else:
            self.status_dot.setStyleSheet("color:#e74c3c;")
            self.status_label.setText("Ollama не найден")
            self.statusBar().showMessage(f"Ошибка: {detail}")
            QMessageBox.warning(
                self,
                "Ollama не запущен",
                "Не удалось подключиться к Ollama.\n\n"
                "Запустите в терминале:\n"
                "  set OLLAMA_MODELS=E:\\ia\\manifests\\registry.ollama.ai\\library\n"
                "  ollama serve"
            )

    def _on_stream_finished(self):
        self.is_generating = False
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        self.statusBar().showMessage("Готово")

    def _on_error(self, msg):
        self.is_generating = False
        self.send_btn.setEnabled(True)
        self._append_html(
            f'<div style="margin:8px 0;color:#e74c3c;">'
            f'⚠️ Ошибка: {self._escape(msg)}</div>'
        )

    # ──────────────────────────────────────────
    #  Утилиты
    # ──────────────────────────────────────────
    def _append_html(self, html):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    @staticmethod
    def _escape(text):
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _clear_chat(self):
        self.chat_display.clear()
        self.conversation_history.clear()
        self.statusBar().showMessage("История очищена")


# ──────────────────────────────────────────────
#  Точка входа
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = GPTOSSChatWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()