import sys
from ultralytics import YOLO
from PySide6 import QtCore, QtWidgets, QtGui
from PIL import Image
from MangaOCRModel import MangaOCRModel


class MessageDialog(QtWidgets.QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)

        layout = QtWidgets.QVBoxLayout(self)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText(str(message))
        text_edit.setReadOnly(True)
        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.Monospace)
        text_edit.setFont(font)

        layout.addWidget(text_edit)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
class Bubble:
    def __init__(self, polygon, bbox):
        self.polygon = polygon
        self.bbox = bbox
        self.text_ocr = ""
        self.text_translated = ""


class AppData:
    def __init__(self):
        self.image_path = None
        self.pil_image = None
        self.bubbles = []


class SegmentBubbleTab(QtWidgets.QWidget):
    """
    Class for "Segment Bubble" tab.
    Load Image, Select Model, Run Inference
    """

    def __init__(self, data_context):
        super().__init__()
        self.data = data_context

        self.model = None
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
        self.layout = QtWidgets.QHBoxLayout(self)

        # --- left --- #
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        left_layout.addWidget(self.selectModelButton)
        left_layout.addWidget(self.openImageButton)
        left_layout.addWidget(self.imageList)

        # --- right --- #
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        inference_layout = QtWidgets.QHBoxLayout()
        inference_layout.addWidget(self.confidenceLabel)
        inference_layout.addWidget(self.confidenceSlider)
        inference_layout.addWidget(self.runInferenceButton)
        right_layout.addLayout(inference_layout)

        right_layout.addWidget(self.view)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.zoomInButton)
        bottom_layout.addWidget(self.zoomOutButton)
        bottom_layout.addWidget(self.fitButton)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.deleteButton)
        right_layout.addLayout(bottom_layout)

        # --- assemble --- #
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.splitter)

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
            self.data.image_path = file_path
            try:
                self.data.pil_image = Image.open(file_path)
            except Exception as e:
                print(f"Error loading PIL image: {e}")

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
                MessageDialog("Error", f"Could not load model: {e}", self).exec()

    @QtCore.Slot()
    def runInference(self):
        if not self.model or not self.image_path:
            return

        try:
            for item in self.scene.items():
                if item != self.pixmap_item:
                    self.scene.removeItem(item)

            conf_val = self.confidenceSlider.value() / 100.0

            results = self.model(self.image_path, conf=conf_val)
            result = results[0]

            if result.masks is None:
                print("No segmentation masks found.")
                return

            masks = result.masks.xy
            boxes = result.boxes.xyxy.cpu().numpy()

            for segment_points, box in zip(masks, boxes):
                polygon = QtGui.QPolygonF()
                for point in segment_points:
                    polygon.append(QtCore.QPointF(point[0], point[1]))

                bubble = Bubble(polygon, box.tolist())
                self.data.bubbles.append(bubble)

                poly_item = QtWidgets.QGraphicsPolygonItem(polygon)

                pen = QtGui.QPen(QtCore.Qt.red)
                pen.setWidth(2)
                poly_item.setPen(pen)
                poly_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))

                poly_item.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable)

                poly_item.setData(0, bubble)

                self.scene.addItem(poly_item)

            print(f"Found {len(result.masks.xy)} segmented bubbles")

        except Exception as e:
            print(f"Inference Error: {e}")
            MessageDialog("Error", f"Inference failed: \n{e}", self).exec()

    @QtCore.Slot()
    def deleteSelectedBubble(self):
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if item != self.pixmap_item:
                bubble_ref = item.data(0)
                if bubble_ref in self.data.bubbles:
                    self.data.bubbles.remove(bubble_ref)
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


class OCRTab(QtWidgets.QWidget):
    def __init__(self, data_context):
        super().__init__()
        self.data = data_context

        self.ocr_model = MangaOCRModel()
        self.model_loaded = False

        # --- buttons --- #
        self.loadModelButton = QtWidgets.QPushButton("Load OCR Model")
        self.runOCRButton = QtWidgets.QPushButton("Run OCR")
        self.resultList = QtWidgets.QListWidget()

        self.zoomInButton = QtWidgets.QPushButton("Zoom In (+)")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom Out (-)")
        self.fitButton = QtWidgets.QPushButton("Fit to space")

        # --- layout --- #
        self.layout = QtWidgets.QHBoxLayout(self)

        # --- left --- #
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        left_layout.addWidget(self.loadModelButton)
        left_layout.addWidget(self.runOCRButton)
        left_layout.addWidget(QtWidgets.QLabel("Detected Text:"))
        left_layout.addWidget(self.resultList)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)

        # --- right --- #
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        right_layout.addWidget(self.view)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(self.zoomInButton)
        zoom_layout.addWidget(self.zoomOutButton)
        zoom_layout.addWidget(self.fitButton)
        zoom_layout.addStretch()
        # TODO: change this to edit button
        # zoom_layout.addWidget(self.deleteButton)
        right_layout.addLayout(zoom_layout)

        # --- assemble --- #
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)

        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.splitter)

        # --- signal and slot connection --- #
        self.loadModelButton.clicked.connect(self.loadModel)
        self.runOCRButton.clicked.connect(self.runOCR)
        self.resultList.itemClicked.connect(self.highlightBubble)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.fitButton.clicked.connect(self.fitView)

    @QtCore.Slot()
    def loadModel(self):
        try:
            self.loadModelButton.setText("Loading...")
            self.loadModelButton.setEnabled(False)
            QtWidgets.QApplication.processEvents()

            self.ocr_model.load_model()
            self.model_loaded = True

            self.loadModelButton.setText("Model Loaded")
            QtWidgets.QMessageBox.information(self, "success", "MangaOCR model loaded.")
        except Exception as e:
            self.loadModelButton.setText("Load OCR model")
            self.loadModelButton.setEnabled(True)
            MessageDialog("Error", f"Failed to load OCR model: {e}", self).exec()

    @QtCore.Slot()
    def runOCR(self):
        if not self.model_loaded:
            QtWidgets.QMessageBox.warning(
                self, "warning", "please load the OCR model first."
            )
            return
        if not self.data.pil_image or not self.data.bubbles:
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "No image or segmented bubbles found. Please go to Segment tab.",
            )
            return

        self.scene.clear()
        pixmap = QtGui.QPixmap(self.data.image_path)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        bboxes = []
        valid_bubbles = []

        for bubble in self.data.bubbles:
            bboxes.append(bubble.bbox)
            valid_bubbles.append(bubble)

        if not bboxes:
            return

        try:
            texts = self.ocr_model.predict(self.data.pil_image, bboxes)

            self.resultList.clear()

            for i, text in enumerate(texts):
                valid_bubbles[i].text_ocr = text

                item = QtWidgets.QListWidgetItem(f"{i + 1}: {text}")
                item.setData(QtCore.Qt.UserRole, valid_bubbles[i])
                self.resultList.addItem(item)

                x1, y1, x2, y2 = valid_bubbles[i].bbox
                w = x2 - x1
                h = y2 - y1
                rect = QtCore.QRectF(x1, y1, w, h)

                rect_item = QtWidgets.QGraphicsRectItem(rect)
                rect_item.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                self.scene.addItem(rect_item)

                text_item = QtWidgets.QGraphicsTextItem(text)
                text_item.setDefaultTextColor(QtCore.Qt.blue)
                text_item.setPos(x1, y1)
                text_item.setScale(1.5)
                self.scene.addItem(text_item)

        except Exception as e:
            MessageDialog("Error", f"OCR Failed: {e}", self).exec()

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def highlightBubble(self, item):
        bubble = item.data(QtCore.Qt.UserRole)
        if bubble:
            x1, y1, x2, y2 = bubble.bbox
            rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
            self.view.ensureVisible(rect)

    @QtCore.Slot()
    def zoomIn(self):
        self.view.scale(1.2, 1.2)

    @QtCore.Slot()
    def zoomOut(self):
        self.view.scale(0.8, 0.8)

    @QtCore.Slot()
    def fitView(self):
        if self.data.image_path:
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
class MainApplication(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lucile")

        self.app_data = AppData()

        # --- main layout --- #
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # --- tab widgets --- #
        self.tabs = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # --- actual tabs --- #
        self.segment_tab = SegmentBubbleTab(self.app_data)
        self.tabs.addTab(self.segment_tab, "Segment bubble")

        self.ocr_tab = OCRTab(self.app_data)
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
