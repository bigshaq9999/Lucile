import sys
import os
from PySide6 import QtCore, QtWidgets, QtGui
from PIL import Image
import cv2
import numpy as np

from Lucile_bigshaq9999.MangaTypesetter import MangaTypesetter
from Lucile_bigshaq9999.BubbleSegmenter import BubbleSegmenter


class SegmentLoaderWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            from huggingface_hub import hf_hub_download

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            file_path = hf_hub_download(
                repo_id=f"TheBlindMaster/{self.model_name}-manga-bubble-seg",
                filename="best.pt",
                local_dir=os.path.join(
                    BASE_DIR,
                    "models",
                    "bubble-detection",
                    "YOLO_Ultralytics",
                    self.model_name,
                ),
                local_dir_use_symlinks=False,
            )

            if file_path:
                model = BubbleSegmenter(file_path)
                self.finished.emit(model)
            else:
                self.error.emit("Could not download or locate model file.")

        except Exception as e:
            self.error.emit(str(e))


class OCRLoaderWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def run(self):
        try:
            from Lucile_bigshaq9999.MangaOCRModel import MangaOCRModel

            model_wrapper = MangaOCRModel()
            model_wrapper.load_model()

            self.finished.emit(model_wrapper)

        except Exception as e:
            self.error.emit(str(e))


class TranslatorLoadWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(self, model_size):
        super().__init__()
        self.model_size = model_size

    def run(self):
        try:
            from Lucile_bigshaq9999.ElanMtJaEnTranslator import ElanMtJaEnTranslator

            translator = ElanMtJaEnTranslator()

            translator.load_model(device="auto", elan_model=self.model_size)

            self.finished.emit(translator)

        except Exception as e:
            self.error.emit(str(e))


class InferenceWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        try:
            image_rgb, raw_masks, refined_bubbles = self.model.detect_and_segment(
                self.image_path
            )
            self.finished.emit(image_rgb, raw_masks, refined_bubbles)
        except Exception as e:
            self.error.emit(str(e))


class OCRWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    error = QtCore.Signal(str)

    def __init__(self, ocr_model, pil_image, bboxes, valid_bubbles):
        super().__init__()
        self.ocr_model = ocr_model
        self.pil_image = pil_image
        self.bboxes = bboxes
        self.valid_bubbles = valid_bubbles

    def run(self):
        try:
            texts = self.ocr_model.predict(self.pil_image, self.bboxes)
            self.finished.emit(texts, self.valid_bubbles)
        except Exception as e:
            self.error.emit(str(e))


class TranslationWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object, object)
    error = QtCore.Signal(str)

    def __init__(self, translator, bubbles_to_translate, source_texts):
        super().__init__()
        self.translator = translator
        self.bubbles_to_translate = bubbles_to_translate
        self.source_texts = source_texts

    def run(self):
        try:
            translated_texts = self.translator.predict(self.source_texts)
            self.finished.emit(
                self.bubbles_to_translate, self.source_texts, translated_texts
            )
        except Exception as e:
            self.error.emit(str(e))


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
    def __init__(self, polygon, bbox, raw_mask):
        self.polygon = polygon
        self.bbox = bbox
        self.raw_mask = raw_mask
        self.text_ocr = ""
        self.text_translated = ""


class AppData:
    def __init__(self):
        self.image_path = None
        self.pil_image = None
        self.image_np = None
        self.bubbles = []


class SegmentBubbleTab(QtWidgets.QWidget):
    """
    Class for "Segment Bubble" tab.
    Load Image, Select Model, Run Inference
    """

    def __init__(self, data_context: AppData):
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
        self.imageList = QtWidgets.QListWidget()

        self.modelSelector = QtWidgets.QComboBox()
        self.modelSelector.addItems(["yolov8s", "yolov8n", "yolov11s", "yolov11n"])

        self.deleteButton = QtWidgets.QPushButton("Delete Selected Bubble")
        self.deleteButton.setShortcut(QtGui.QKeySequence.Delete)

        self.zoomInButton = QtWidgets.QPushButton("Zoom In (+)")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom Out (-)")
        self.fitButton = QtWidgets.QPushButton("Fit to space")

        # --- layout --- #
        self.layout = QtWidgets.QHBoxLayout(self)

        # --- left --- #
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        left_layout.addWidget(self.modelSelector)
        left_layout.addWidget(self.selectModelButton)
        left_layout.addWidget(self.openImageButton)
        left_layout.addWidget(self.runInferenceButton)
        left_layout.addWidget(self.imageList)

        # --- right --- #
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

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
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.fitButton.clicked.connect(self.fitView)

    @QtCore.Slot()
    def openImage(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.data.bubbles.clear()
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
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    @QtCore.Slot()
    def selectModel(self):
        selected_model_name = self.modelSelector.currentText()

        self.selectModelButton.setText("Loading...")
        self.selectModelButton.setEnabled(True)
        self.modelSelector.setEnabled(True)

        self.thread = QtCore.QThread()
        self.worker = SegmentLoaderWorker(selected_model_name)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.onLoadFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.error.connect(self.onLoadError)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(object)
    def onLoadFinished(self, loaded_model):
        self.model = loaded_model

        self.selectModelButton.setText("Select model")
        self.selectModelButton.setEnabled(True)
        self.modelSelector.setEnabled(True)

        QtWidgets.QMessageBox.information(self, "Success", "Segmentation model loaded.")

    @QtCore.Slot(str)
    def onLoadError(self, error_msg):
        self.selectModelButton.setText("Select model")
        self.selectModelButton.setEnabled(True)
        self.modelSelector.setEnabled(True)

        MessageDialog("Error", f"Could not load model: {error_msg}", self).exec()

    @QtCore.Slot()
    def runInference(self):
        if not self.model:
            QtWidgets.QMessageBox.information(self, "Warning", "Load model first.")
            return

        if not self.image_path:
            QtWidgets.QMessageBox.information(self, "Warning", "Load image first.")
            return

        for item in self.scene.items():
            if item != self.pixmap_item:
                self.scene.removeItem(item)

        self.data.bubbles.clear()

        self.runInferenceButton.setEnabled(False)
        self.runInferenceButton.setText("Running...")

        self.thread = QtCore.QThread()
        self.worker = InferenceWorker(self.model, self.image_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.onInferenceFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.error.connect(self.onInferenceError)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(object, object, object)
    def onInferenceFinished(self, image_rgb, raw_masks, refined_bubbles):
        self.runInferenceButton.setEnabled(True)
        self.runInferenceButton.setText("Run inference")

        self.data.image_np = image_rgb

        if not refined_bubbles:
            QtWidgets.QMessageBox.information(self, "Warning", "No bubble found.")
            return

        print(f"Found {len(refined_bubbles)} segmented bubbles")

        for bubble_data in refined_bubbles:
            contour = bubble_data["contour"]
            polygon = QtGui.QPolygonF()

            for point_wrapper in contour:
                x, y = point_wrapper[0]
                polygon.append(QtCore.QPointF(float(x), float(y)))

            x, y, w, h = bubble_data["bbox"]
            box_xyxy = [x, y, x + w, y + h]

            raw_mask_np = bubble_data["original_mask"]
            bubble_obj = Bubble(polygon, box_xyxy, raw_mask_np)
            self.data.bubbles.append(bubble_obj)

            poly_item = QtWidgets.QGraphicsPolygonItem(polygon)
            pen = QtGui.QPen(QtCore.Qt.red)
            pen.setWidth(2)
            poly_item.setPen(pen)
            poly_item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))
            poly_item.setFlags(QtWidgets.QGraphicsItem.ItemIsSelectable)
            poly_item.setData(0, bubble_obj)

            self.scene.addItem(poly_item)

    @QtCore.Slot(str)
    def onInferenceError(self, error_msg):
        self.runInferenceButton.setEnabled(True)
        self.runInferenceButton.setText("Run inference")
        print(f"Inference Error: {error_msg}")
        MessageDialog("Error", f"Inference failed: \n{error_msg}", self).exec()

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
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


class OCRTab(QtWidgets.QWidget):
    def __init__(self, data_context: AppData):
        super().__init__()
        self.data = data_context
        self.selected_bubble = None

        self.ocr_model = None
        self.model_loaded = False

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # --- buttons --- #
        self.loadModelButton = QtWidgets.QPushButton("Load OCR Model")
        self.runOCRButton = QtWidgets.QPushButton("Run OCR")
        self.resultList = QtWidgets.QListWidget()

        # --- edit text --- #
        self.textEditor = QtWidgets.QTextEdit()
        self.textEditor.setPlaceholderText("Select a bubble to edit text.")

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
        left_layout.addWidget(QtWidgets.QLabel("Bubbles:"))
        left_layout.addWidget(self.resultList, 1)
        left_layout.addWidget(QtWidgets.QLabel("Edit text:"))
        left_layout.addWidget(self.textEditor, 1)

        # --- right --- #
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        right_layout.addWidget(self.view)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(self.zoomInButton)
        zoom_layout.addWidget(self.zoomOutButton)
        zoom_layout.addWidget(self.fitButton)
        zoom_layout.addStretch()
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
        self.textEditor.textChanged.connect(self.updateBubbleText)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.fitButton.clicked.connect(self.fitView)

    @QtCore.Slot()
    # TODO: What if I want to load another model? It's annoying to load
    # the model everytime I want to use the app.
    def loadModel(self):
        self.loadModelButton.setText("Loading...")
        self.loadModelButton.setEnabled(False)

        self.thread = QtCore.QThread()
        self.worker = OCRLoaderWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.onLoadFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.error.connect(self.onLoadError)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(object)
    def onLoadFinished(self, loaded_model):
        self.ocr_model = loaded_model
        self.model_loaded = True

        self.loadModelButton.setText("Model loaded")
        self.loadModelButton.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Success", "MangaOCR model loaded.")

    @QtCore.Slot(str)
    def onLoadError(self, error_msg):
        self.loadModelButton.setText("Load OCR model")
        self.loadModelButton.setEnabled(True)
        MessageDialog("Error", f"Failed to load model:\n{error_msg}", self).exec()

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

        bboxes = []
        valid_bubbles = []

        for bubble in self.data.bubbles:
            bboxes.append(bubble.bbox)
            valid_bubbles.append(bubble)

        if not bboxes:
            return

        self.runOCRButton.setEnabled(False)
        self.runOCRButton.setText("Loading...")
        self.resultList.clear()

        self.scene.clear()

        self.thread = QtCore.QThread()
        self.worker = OCRWorker(
            self.ocr_model, self.data.pil_image, bboxes, valid_bubbles
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.onOCRFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.error.connect(self.onOCRError)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(object, object)
    def onOCRFinished(self, texts, valid_bubbles):
        self.runOCRButton.setEnabled(True)
        self.runOCRButton.setText("Loading...")

        self.scene.clear()
        pixmap = QtGui.QPixmap(self.data.image_path)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        for i, text in enumerate(texts):
            valid_bubbles[i].text_ocr = text

            preview = text[:20] + "..." if len(text) > 20 else text
            item = QtWidgets.QListWidgetItem(f"{i + 1}: {preview}")
            item.setData(QtCore.Qt.UserRole, valid_bubbles[i])
            self.resultList.addItem(item)

            x1, y1, x2, y2 = valid_bubbles[i].bbox
            w = x2 - x1
            h = y2 - y1
            rect = QtCore.QRectF(x1, y1, w, h)

            rect_item = QtWidgets.QGraphicsRectItem(rect)
            rect_item.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
            self.scene.addItem(rect_item)

            # TODO: option to change color
            text_item = QtWidgets.QGraphicsTextItem(text)
            text_item.setDefaultTextColor(QtCore.Qt.blue)
            text_item.setPos(x1, y1)
            text_item.setScale(1.5)
            self.scene.addItem(text_item)

            valid_bubbles[i].graphics_item = text_item

    @QtCore.Slot(str)
    def onOCRError(self, error_msg):
        self.runOCRButton.setEnabled(True)
        self.runOCRButton.setText("Run OCR")
        MessageDialog("Error", f"OCR Failed: {error_msg}", self).exec()

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def highlightBubble(self, item):
        bubble = item.data(QtCore.Qt.UserRole)
        self.selected_bubble = bubble

        if bubble:
            self.textEditor.blockSignals(True)
            self.textEditor.setPlainText(bubble.text_ocr)
            self.textEditor.blockSignals(False)

            x1, y1, x2, y2 = bubble.bbox
            w = x2 - x1
            h = y2 - y1
            rect = QtCore.QRectF(x1, y1, w, h)
            self.view.ensureVisible(rect)

    @QtCore.Slot()
    def updateBubbleText(self):
        if self.selected_bubble:
            new_text = self.textEditor.toPlainText()
            self.selected_bubble.text_ocr = new_text

            if (
                hasattr(self.selected_bubble, "graphics_item")
                and self.selected_bubble.graphics_item
            ):
                self.selected_bubble.graphics_item.setPlainText(new_text)

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


class TranslateTab(QtWidgets.QWidget):
    def __init__(self, data_context: AppData):
        super().__init__()
        self.data = data_context
        self.selected_bubble = None

        self.translator = None
        self.model_loaded = False

        # --- graphics view --- #
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # --- controls --- #
        self.modelSelector = QtWidgets.QComboBox()
        self.modelSelector.addItems(["tiny", "base", "bt"])
        self.modelSelector.setToolTip(
            "Select model size: Tiny (fast), Base, BT (Better quality)"
        )

        self.loadModelButton = QtWidgets.QPushButton("Load translator")
        self.runTranslateButton = QtWidgets.QPushButton("Run translation")
        self.resultList = QtWidgets.QListWidget()

        # --- editor --- #
        self.originalTextEdit = QtWidgets.QTextEdit()
        self.originalTextEdit.setReadOnly(True)

        self.originalTextEdit.setMaximumHeight(100)

        self.transalatedTextEdit = QtWidgets.QTextEdit()
        self.transalatedTextEdit.setPlaceholderText("Edit translation here...")

        self.zoomInButton = QtWidgets.QPushButton("Zoom in (+)")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom out (-)")
        self.fitButton = QtWidgets.QPushButton("Fit to space")

        # --- layout --- #
        self.layout = QtWidgets.QHBoxLayout(self)

        # --- left --- #
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        left_layout.addWidget(QtWidgets.QLabel("Model size:"))
        left_layout.addWidget(self.modelSelector)
        left_layout.addWidget(self.loadModelButton)
        left_layout.addWidget(self.runTranslateButton)

        left_layout.addWidget(QtWidgets.QLabel("Select bubble:"))
        left_layout.addWidget(self.resultList, 1)

        left_layout.addWidget(QtWidgets.QLabel("Original (JP):"))
        left_layout.addWidget(self.originalTextEdit)

        left_layout.addWidget(QtWidgets.QLabel("Translation (EN):"))
        left_layout.addWidget(self.transalatedTextEdit, 1)

        # --- right --- #
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        right_layout.addWidget(self.view)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(self.zoomInButton)
        zoom_layout.addWidget(self.zoomOutButton)
        zoom_layout.addWidget(self.fitButton)
        zoom_layout.addStretch()
        right_layout.addLayout(zoom_layout)

        # --- assemble --- #
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.splitter)

        # --- connection --- #
        self.loadModelButton.clicked.connect(self.loadModel)
        self.runTranslateButton.clicked.connect(self.runTranslation)
        self.resultList.itemClicked.connect(self.highlightBubble)
        self.transalatedTextEdit.textChanged.connect(self.updateTranslation)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.fitButton.clicked.connect(self.fitView)

    @QtCore.Slot()
    def loadModel(self):
        # TODO: make it possible to change model after loading
        selected_model = self.modelSelector.currentText()

        self.loadModelButton.setText("Loading...")
        self.loadModelButton.setEnabled(False)
        self.modelSelector.setEnabled(False)

        self.thread = QtCore.QThread()
        self.worker = TranslatorLoadWorker(selected_model)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.onLoadFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.error.connect(self.onLoadError)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(object)
    def onLoadFinished(self, loaded_translator):
        self.translator = loaded_translator
        self.model_loaded = True

        self.loadModelButton.setText("Ready")
        self.loadModelButton.setEnabled(True)
        self.modelSelector.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Success", "Translator model loaded.")

    @QtCore.Slot(str)
    def onLoadError(self, error_msg):
        self.loadModelButton.setText("Load translator")
        self.loadModelButton.setEnabled(True)
        self.modelSelector.setEnabled(True)
        MessageDialog("Error", f"Failed to load translator: {error_msg}", self).exec()

    @QtCore.Slot()
    def runTranslation(self):
        if not self.model_loaded or self.translator is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Load the translator model first."
            )
            return

        bubbles_to_translate = [b for b in self.data.bubbles if b.text_ocr]

        if not bubbles_to_translate:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No OCR text found. Run OCR first."
            )
            return

        source_texts = [b.text_ocr for b in bubbles_to_translate]

        self.runTranslateButton.setEnabled(False)
        self.runTranslateButton.setText("Translating...")

        self.thread = QtCore.QThread()
        self.worker = TranslationWorker(
            self.translator, bubbles_to_translate, source_texts
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.onTranslationFinished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.worker.error.connect(self.onTranslationError)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @QtCore.Slot(object, object, object)
    def onTranslationFinished(self, bubbles_list, source_list, translated_texts):
        self.runTranslateButton.setEnabled(True)
        self.runTranslateButton.setText("Translate")

        self.resultList.clear()
        self.scene.clear()

        if self.data.image_path:
            pixmap = QtGui.QPixmap(self.data.image_path)
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

            for i, bubble in enumerate(bubbles_list):
                bubble.text_translated = translated_texts[i]

                item = QtWidgets.QListWidgetItem(source_list[i])
                item.setData(QtCore.Qt.UserRole, bubble)
                self.resultList.addItem(item)

                x1, y1, x2, y2 = bubble.bbox
                w = x2 - x1
                h = y2 - y1
                rect = QtCore.QRectF(x1, y1, w, h)

                rect_item = QtWidgets.QGraphicsRectItem(rect)
                rect_item.setPen(QtGui.QPen(QtCore.Qt.green, 2))
                self.scene.addItem(rect_item)

                text_item = QtWidgets.QGraphicsTextItem(bubble.text_translated)
                text_item.setDefaultTextColor(QtCore.Qt.green)
                text_item.setPos(x1, y1)
                text_item.setScale(1.5)
                text_item.setZValue(1)
                self.scene.addItem(text_item)

                bubble.graphics_item_en = text_item

    @QtCore.Slot(str)
    def onTranslationError(self, error_msg):
        self.runTranslateButton.setEnabled(True)
        self.runTranslateButton.setText("Translate")
        MessageDialog("Error", f"Translation failed: {error_msg}", self).exec()

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def highlightBubble(self, item):
        bubble = item.data(QtCore.Qt.UserRole)
        self.selected_bubble = bubble

        if bubble:
            self.originalTextEdit.setText(bubble.text_ocr)

            self.transalatedTextEdit.blockSignals(True)
            self.transalatedTextEdit.setText(bubble.text_translated)
            self.transalatedTextEdit.blockSignals(False)

            x1, y1, x2, y2 = bubble.bbox
            w = x2 - x1
            h = y2 - y1
            rect = QtCore.QRectF(x1, y1, w, h)
            self.view.ensureVisible(rect)

    @QtCore.Slot()
    def updateTranslation(self):
        if self.selected_bubble:
            new_text = self.transalatedTextEdit.toPlainText()
            self.selected_bubble.text_translated = new_text

            if (
                hasattr(self.selected_bubble, "graphics_item_en")
                and self.selected_bubble.graphics_item_en
            ):
                self.selected_bubble.graphics_item_en.setPlainText(new_text)

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


class ReplaceTab(QtWidgets.QWidget):
    def __init__(self, data_context: AppData):
        super().__init__()
        self.data = data_context
        self.typesetter = MangaTypesetter()
        self.final_image_pil = None

        # --- graphics view --- #
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # --- controls --- #
        self.runButton = QtWidgets.QPushButton("Typeset")
        self.saveButton = QtWidgets.QPushButton("Save image")
        self.saveButton.setEnabled(False)

        self.zoomInButton = QtWidgets.QPushButton("Zoom In (+)")
        self.zoomOutButton = QtWidgets.QPushButton("Zoom Out (-)")
        self.fitButton = QtWidgets.QPushButton("Fit to space")

        # --- layout --- #
        self.layout = QtWidgets.QHBoxLayout(self)

        # --- left --- #
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        left_layout.addWidget(QtWidgets.QLabel("Typesetting"))
        left_layout.addSpacing(10)
        left_layout.addWidget(self.runButton)
        left_layout.addWidget(self.saveButton)
        left_layout.addStretch()

        # --- right --- #
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.addWidget(self.view)

        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(self.zoomInButton)
        zoom_layout.addWidget(self.zoomOutButton)
        zoom_layout.addWidget(self.fitButton)
        zoom_layout.addStretch()
        right_layout.addLayout(zoom_layout)

        # --- assemble --- #
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        self.layout.addWidget(self.splitter)

        # --- connections --- #
        self.runButton.clicked.connect(self.runTypesetting)
        self.saveButton.clicked.connect(self.saveImage)
        self.zoomInButton.clicked.connect(self.zoomIn)
        self.zoomOutButton.clicked.connect(self.zoomOut)
        self.fitButton.clicked.connect(self.fitView)

    def _polygon_to_mask(self, polygon, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = []

        for i in range(polygon.count()):
            pt = polygon.at(i)
            points.append([int(pt.x()), int(pt.y())])

        if not points:
            return mask

        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    @QtCore.Slot()
    def runTypesetting(self):
        if self.data.image_np is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        if not self.data.bubbles:
            QtWidgets.QMessageBox.warning(self, "Warning", "No bubbles found.")
            return

        try:
            self.runButton.setText("Processing...")
            self.runButton.setEnabled(False)
            QtWidgets.QApplication.processEvents()

            original_image = self.data.image_np
            height, width, _ = original_image.shape

            formatted_bubbles = []

            for bubble in self.data.bubbles:
                text = (
                    bubble.text_translated
                    if bubble.text_translated
                    else bubble.text_ocr
                )

                refined_mask = self._polygon_to_mask(bubble.polygon, width, height)

                formatted_bubbles.append({
                    "translated_text": text,
                    "mask": refined_mask,
                    "original_mask": bubble.raw_mask,
                })

            final_image_np = self.typesetter.render(original_image, formatted_bubbles)

            self.final_image_pil = Image.fromarray(final_image_np)

            image_data = self.final_image_pil.convert("RGBA").tobytes("raw", "RGBA")
            qim = QtGui.QImage(image_data, width, height, QtGui.QImage.Format_RGBA8888)
            pixmap = QtGui.QPixmap.fromImage(qim)

            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

            self.runButton.setText("Typeset")
            self.runButton.setEnabled(True)
            self.saveButton.setEnabled(True)

        except Exception as e:
            self.runButton.setText("Typeset")
            self.runButton.setEnabled(True)
            MessageDialog("Error", f"Typesetting failed: {e}", self).exec()

    @QtCore.Slot()
    def saveImage(self):
        if not self.final_image_pil:
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", "", "PNG files (*.png);;JPEG files (*.jpg)"
        )

        if file_path:
            try:
                self.final_image_pil.save(file_path)
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Image saved to {file_path}"
                )
            except Exception as e:
                MessageDialog("Error", f"Failed to save image: {e}", self).exec()

    @QtCore.Slot()
    def zoomIn(self):
        self.view.scale(1.2, 1.2)

    @QtCore.Slot()
    def zoomOut(self):
        self.view.scale(0.8, 0.8)

    @QtCore.Slot()
    def fitView(self):
        if self.scene.items():
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

        self.translate_tab = TranslateTab(self.app_data)
        self.tabs.addTab(self.translate_tab, "Translate")

        self.replace_tab = ReplaceTab(self.app_data)
        self.tabs.addTab(self.replace_tab, "Replace")


def main():
    app = QtWidgets.QApplication([])

    widget = MainApplication()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
