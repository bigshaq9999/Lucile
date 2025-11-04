import sys
from PySide6 import QtCore, QtWidgets, QtGui
from ultralytics import YOLO


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.model = None
        self.original_pixmap = None

        self.image_label = QtWidgets.QLabel("No image selected")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.image_label.setMinimumSize(6, 10)

        # --- buttons creation --- #
        self.openImageButton = QtWidgets.QPushButton("Open Image")
        self.selectModelButton = QtWidgets.QPushButton("Select Model")
        self.runInferenceButton = QtWidgets.QPushButton("Run Inference")

        # --- layout --- #
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.image_label)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.selectModelButton)
        button_layout.addWidget(self.openImageButton)
        button_layout.addWidget(self.runInferenceButton)

        self.layout.addLayout(button_layout)

        # --- signal and slot connection --- #
        self.selectModelButton.clicked.connect(self.selectModel)
        self.openImageButton.clicked.connect(self.openImage)
        self.runInferenceButton.clicked.connect(self.runInference)

    def update_image_display(self):
        if self.original_pixmap is None:
            return

        label_size = self.image_label.size()

        scaled_pixmap = self.original_pixmap.scaled(
            label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()

    @QtCore.Slot()
    def openImage(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        self.image_path = file_path

        if file_path:
            self.original_pixmap = QtGui.QPixmap(file_path)
            self.update_image_display()

    @QtCore.Slot()
    def selectModel(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Files (*.pt)"
        )

        if file_path:
            try:
                self.model = YOLO(file_path)
                print(f"Model loaded successfully from {file_path}")
            except Exception as e:
                print(f"Error loading model: {e}")

    @QtCore.Slot()
    def runInference(self):
        if not self.model:
            print("Please select a model first.")
            return
        if not self.image_path:
            print("Please open an image first.")
            return
        try:
            results = self.model(self.image_path)
            annotated_image = results[0].plot()

            height, width, channel = annotated_image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(
                annotated_image.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format_BGR888,
            )

            self.original_pixmap = QtGui.QPixmap.fromImage(q_image)
            self.update_image_display()

        except Exception as e:
            print(f"An error occured during inference: {e}")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
