import sys
import numpy as np
import sounddevice as sd
from collections import deque
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from typing import Optional, Tuple
import threading

# --- Configurações ---
SAMPLE_RATE = 44100
BUFFER_SIZE = 30000  
UPDATE_INTERVAL_MS = 16  

class AudioProcessor:
    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_size: int = BUFFER_SIZE):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.stream: Optional[sd.InputStream] = None
        
        # Parâmetros de Calibração (Ganhos e Offsets)
        self.gain_x = 1.0
        self.gain_y = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        if status:
            print(f"⚠️ Status do Áudio: {status}", file=sys.stderr)
        
        with self.lock:
            # Aplica calibração em tempo real no processamento vetorial
            x_processed = (indata[:, 0] * self.gain_x) + self.offset_x
            y_processed = (indata[:, 1] * self.gain_y) + self.offset_y
            self.x_buffer.extend(x_processed)
            self.y_buffer.extend(y_processed)

    def start(self):
        try:
            # Tenta abrir o dispositivo padrão
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self._audio_callback,
                blocksize=0
            )
            self.stream.start()
        except Exception as e:
            print(f"\n[ERRO] Falha ao inicializar áudio: {e}")
            print("\nDispositivos de Entrada Disponíveis:")
            print(sd.query_devices())
            sys.exit(1)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            return np.array(self.x_buffer), np.array(self.y_buffer)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, processor: AudioProcessor):
        super().__init__()
        self.processor = processor
        self.init_ui()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(UPDATE_INTERVAL_MS)

    def init_ui(self):
        self.setWindowTitle("Chua Oscilloscope Pro - XY Visualizer")
        self.resize(1000, 900)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Área de Status e Controles
        self.info_label = QtWidgets.QLabel(
            "Teclas: [W/S] Ganho X | [O/L] Ganho Y | [R] Reset | [Espaço] Pausar"
        )
        self.info_label.setStyleSheet("color: #00FF41; font-weight: bold; background: #111;")
        layout.addWidget(self.info_label)

        # Configuração do Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setBackground('#050505')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Efeito Glow/Phosphor
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 255, 65, 200), width=1.5),
            antialias=True
        )
        
        layout.addWidget(self.plot_widget)
        
        self.is_paused = False

    def keyPressEvent(self, event):
        """Controles em tempo real para ajuste de sinal do hardware."""
        step = 1.1
        if event.key() == QtCore.Qt.Key.Key_W: self.processor.gain_x *= step
        elif event.key() == QtCore.Qt.Key.Key_S: self.processor.gain_x /= step
        elif event.key() == QtCore.Qt.Key.Key_O: self.processor.gain_y *= step
        elif event.key() == QtCore.Qt.Key.Key_L: self.processor.gain_y /= step
        elif event.key() == QtCore.Qt.Key.Key_R:
            self.processor.gain_x = 1.0
            self.processor.gain_y = 1.0
        elif event.key() == QtCore.Qt.Key.Key_Space:
            self.is_paused = not self.is_paused
            
        self.update_status_bar()

    def update_status_bar(self):
        status = f"GX: {self.processor.gain_x:.2f} | GY: {self.processor.gain_y:.2f}"
        if self.is_paused: status += " [PAUSADO]"
        self.info_label.setText(status + " | [W/S/O/L] Ajustar Ganhos | [R] Reset")

    def update_plot(self):
        if self.is_paused: return
        
        x, y = self.processor.get_data()
        if len(x) > 0:
            self.curve.setData(x, y)

def main():
    # Aviso de Segurança Obrigatório
    print("\n" + "="*60)
    print("⚠️  AVISO DE SEGURANÇA CRÍTICO  ⚠️")
    print("="*60)
    print("Certifique-se de que o DIVISOR DE TENSÃO está conectado!")
    print("O circuito de Chua opera em ±9V, o que pode DANIFICAR permanentemente")
    print("a entrada de áudio do seu computador.")
    print("Reduza a amplitude do sinal para menos de 1V pico-a-pico.")
    print("="*60 + "\n")

    app = QtWidgets.QApplication(sys.argv)
    
    # Inicializa Processamento de Áudio
    processor = AudioProcessor()
    processor.start()
    
    # Inicializa Interface Gráfica
    window = MainWindow(processor)
    window.show()
    
    try:
        sys.exit(app.exec())
    finally:
        processor.stop()

if __name__ == "__main__":
    main()
