import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
from ultralytics import YOLO
from PySide6 import QtCore, QtWidgets, QtGui


class SegmentBubbleTab(QtWidgets.QWidget):
    """
    Class for "Segment Bubble" tab.
    Load Image, Select Model, Run Inference
    """

    def __init__(self):
        super().__init__()

        self.model = None
        self.image_path = None
        self.pixmap_item = None

        # --- graphics view --- #
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # --- buttons creation --- #
        self.openImageButton = QtWidgets.QPushButton("Open Image")
        self.selectModelButton = QtWidgets.QPushButton("Select Model")
        self.runInferenceButton = QtWidgets.QPushButton("Run Inference")

        self.deleteButton = QtWidgets.QPushButton("Delete Selected Bubble")
        self.deleteButton.setShortcut(QtGui.QKeySequence.Delete)

        self.zoomInButton = QtWidgets.QPushButton("Zoom In (+)")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom Out (-)")
        self.fitButton = QtWidgets.QPushButton("Fit to space")

        # confidence slider
        self.confidenceLabel = QtWidgets.QLabel("Confidence: 0.25")
        self.confidenceSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.confidenceSlider.setMinimum(1)
        self.confidenceSlider.setMaximum(100)
        self.confidenceSlider.setValue(25)
        self.confidenceSlider.setTickInterval(10)

        # --- layout --- #
        self.layout = QtWidgets.QVBoxLayout(self)

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(self.selectModelButton)
        top_layout.addWidget(self.openImageButton)
        self.layout.addLayout(top_layout)

        inf_layout = QtWidgets.QHBoxLayout()
        inf_layout.addWidget(self.confidenceLabel)
        inf_layout.addWidget(self.confidenceSlider)
        inf_layout.addWidget(self.runInferenceButton)
        self.layout.addLayout(inf_layout)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(self.zoomInButton)
        zoom_layout.addWidget(self.zoomOutButton)
        zoom_layout.addWidget(self.fitButton)
        self.layout.addLayout(zoom_layout)

        self.layout.addWidget(self.view)

        edit_layout = QtWidgets.QHBoxLayout()
        edit_layout.addStretch()
        edit_layout.addWidget(self.deleteButton)
        self.layout.addLayout(edit_layout)

        # --- signal and slot connection --- #
        self.selectModelButton.clicked.connect(self.selectModel)
        self.openImageButton.clicked.connect(self.openImage)
        self.runInferenceButton.clicked.connect(self.runInference)
        self.deleteButton.clicked.connect(self.deleteSelectedBubble)
        self.confidenceSlider.valueChanged.connect(self.updateConfLabel)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.fitButton.clicked.connect(self.fitView)

    def updateConfLabel(self, value):
        self.confidenceLabel.setText(f"Confidence: {value / 100.0:.2f}")

    @QtCore.Slot()
    def openImage(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.image_path = file_path
            self.scene.clear()
            pixmap = QtGui.QPixmap(file_path)
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.view.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)

    @QtCore.Slot()
    def selectModel(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Model", "", "PyTorch Files (*.pt)"
        )

        if file_path:
            try:
                self.model = YOLO(file_path)
                print(f"Model loaded successfully from {file_path}")
                QtWidgets.QMessageBox.information(self, "Success", "Model Loaded.")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Could not load model: {e}"
                )

    @QtCore.Slot()
    def runInference(self):
        if not self.model or not self.image_path:
            return

        try:
            for item in self.scene.items():
                if item != self.pixmap_item:
                    self.scene.removeItem(item)

            conf_val = self.confidenceSlider.value() / 100.0

            results = self.model(self.image_path, conf=conf_val, workers=0)
            result = results[0]

            if result.masks is None:
                print("No segmentation masks found.")
                return

            for segment_points in result.masks.xy:
                polygon = QtGui.QPolygonF()
                for point in segment_points:
                    polygon.append(QtCore.QPointF(point[0], point[1]))

                poly_item = QtWidgets.QGraphicsPolygonItem(polygon)

                pen = QtGui.QPen(QtCore.Qt.red)
                pen.setWidth(2)
                poly_item.setPen(pen)
                poly_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))

                poly_item.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable)
                self.scene.addItem(poly_item)

            print(f"Found {len(result.masks.xy)} segmented bubbles")

        except Exception as e:
            print(f"Inference Error: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Inference failed: \n{e}")

    @QtCore.Slot()
    def deleteSelectedBubble(self):
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if item != self.pixmap_item:
                self.scene.removeItem(item)

    @QtCore.Slot()
    def zoomIn(self):
        self.view.scale(1.2, 1.2)

    @QtCore.Slot()
    def zoomOut(self):
        self.view.scale(0.8, 0.8)

    @QtCore.Slot()
    def fitView(self):
        if self.pixmap_item:
            self.view.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)


class MainApplication(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lucile")

        # --- main layout --- #
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # --- tab widgets --- #
        self.tabs = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # --- actual tabs --- #
        self.segment_tab = SegmentBubbleTab()
        self.tabs.addTab(self.segment_tab, "Segment bubble")

        self.ocr_tab = QtWidgets.QLabel("OCR")
        self.ocr_tab.setAlignment(QtCore.Qt.AlignCenter)
        self.tabs.addTab(self.ocr_tab, "OCR")

        self.translate_tab = QtWidgets.QLabel("Translate")
        self.translate_tab.setAlignment(QtCore.Qt.AlignCenter)
        self.tabs.addTab(self.translate_tab, "Translate")

        self.edit_tab = QtWidgets.QLabel("Edit")
        self.edit_tab.setAlignment(QtCore.Qt.AlignCenter)
        self.tabs.addTab(self.edit_tab, "Edit")

        self.replace_tab = QtWidgets.QLabel("Replace")
        self.replace_tab.setAlignment(QtCore.Qt.AlignCenter)
        self.tabs.addTab(self.replace_tab, "Replace")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MainApplication()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
