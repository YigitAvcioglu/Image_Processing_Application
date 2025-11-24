import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QGroupBox,
                             QScrollArea, QMessageBox, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QListWidget, QListWidgetItem, QDialog, QAbstractItemView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- GÖRÜNTÜ İŞLEME MODÜLÜNÜ DAHİL ET ---
from image_process import ImageEngine


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIP Lab - Checkbox Edition")
        self.setGeometry(100, 100, 1400, 900)

        # --- Tasarım (CSS) ---
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ddd; }
            QLabel { color: #eee; font-size: 13px; }
            QGroupBox { border: 1px solid #444; margin-top: 10px; font-weight: bold; color: #bbb; border-radius: 5px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 6px; border-radius: 4px; min-height: 25px; }
            QPushButton:hover { background-color: #005f9e; }
            QPushButton#DeleteBtn { background-color: #d9534f; }
            QPushButton#DeleteBtn:hover { background-color: #c9302c; }
            QPushButton#SmallReset { background-color: #444; width: 30px; font-weight: bold; border: 1px solid #555; }
            QPushButton#SmallReset:hover { background-color: #d9534f; border: 1px solid #d9534f; }
            QTabWidget::pane { border: 1px solid #333; background: #252526; }
            QTabBar::tab { background: #333; color: #ddd; padding: 8px 15px; }
            QTabBar::tab:selected { background: #007acc; }
            QListWidget { background-color: #252526; border: 1px solid #444; color: #ddd; font-size: 13px; padding: 5px; }
            QListWidget::item { padding: 5px; border-bottom: 1px solid #333; }
            QListWidget::item:selected { background-color: #333; } /* Seçim rengini nötrledik */
            QSlider::handle:horizontal { background: #007acc; width: 16px; margin: -6px 0; border-radius: 8px; }
        """)

        # --- Değişkenler ---
        self.original_image = None
        self.current_image = None
        self.operation_stack = []
        self.is_programmatic_update = False

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- SOL PANEL ---
        left_panel = QWidget()
        left_panel.setFixedWidth(453)
        left_layout = QVBoxLayout(left_panel)

        # Dosya İşlemleri
        file_group = QGroupBox("Dosya")
        file_layout = QHBoxLayout()
        btn_open = QPushButton("Seç");
        btn_open.clicked.connect(self.open_image)
        btn_save = QPushButton("Kaydet");
        btn_save.clicked.connect(self.save_image)
        btn_reset = QPushButton("Sıfırla");
        btn_reset.clicked.connect(self.reset_all)
        btn_close = QPushButton("Kapat")
        btn_close.setToolTip("Resmi Kapat")
        btn_close.setStyleSheet("background-color: #d9534f; font-weight: bold;")
        btn_close.clicked.connect(self.close_image)

        file_layout.addWidget(btn_open);
        file_layout.addWidget(btn_save)
        file_layout.addWidget(btn_reset);
        file_layout.addWidget(btn_close)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Sekmeler
        self.tabs = QTabWidget()

        # --- TAB 1: Fundamentals ---
        tab_fund = QWidget();
        fund_layout = QVBoxLayout(tab_fund)
        h_fund = QHBoxLayout()
        h_fund.addWidget(self.create_btn("Gri Yap", lambda: self.add_op("grayscale", "Grayscale")))
        h_fund.addWidget(self.create_btn("R", lambda: self.reset_op_by_action("grayscale"), "SmallReset"))
        fund_layout.addLayout(h_fund)

        h_neg = QHBoxLayout()
        h_neg.addWidget(self.create_btn("Negatif", lambda: self.add_op("negative", "Negative")))
        h_neg.addWidget(self.create_btn("R", lambda: self.reset_op_by_action("negative"), "SmallReset"))
        fund_layout.addLayout(h_neg)

        h_log = QHBoxLayout()
        h_log.addWidget(self.create_btn("Log Dönüşümü", lambda: self.add_op("log", "Log Transform")))
        h_log.addWidget(self.create_btn("R", lambda: self.reset_op_by_action("log"), "SmallReset"))
        fund_layout.addLayout(h_log)

        group_bp = QGroupBox("Bit-Plane Slicing")
        bp_layout = QHBoxLayout()
        self.slider_bp = QSlider(Qt.Horizontal);
        self.slider_bp.setRange(0, 7);
        self.slider_bp.setValue(7)
        self.lbl_bp = QLabel("7")
        self.slider_bp.valueChanged.connect(
            lambda v: (self.lbl_bp.setText(str(v)), self.add_op("bit_plane", f"Bit-Plane {v}", {"plane": v})))
        btn_rst_bp = QPushButton("R");
        btn_rst_bp.setObjectName("SmallReset")
        btn_rst_bp.clicked.connect(lambda: self.safe_reset_control(self.slider_bp, 7, "bit_plane"))
        bp_layout.addWidget(QLabel("Bit:"));
        bp_layout.addWidget(self.slider_bp);
        bp_layout.addWidget(self.lbl_bp);
        bp_layout.addWidget(btn_rst_bp)
        group_bp.setLayout(bp_layout);
        fund_layout.addWidget(group_bp)

        group_quant = QGroupBox("Quantization")
        quant_layout = QHBoxLayout()
        self.spin_quant = QSpinBox();
        self.spin_quant.setRange(2, 128);
        self.spin_quant.setValue(4)
        self.spin_quant.valueChanged.connect(lambda v: self.add_op("quantize", f"Quantize ({v})", {"levels": v}))
        btn_rst_quant = QPushButton("R");
        btn_rst_quant.setObjectName("SmallReset")
        btn_rst_quant.clicked.connect(lambda: self.safe_reset_control(self.spin_quant, 4, "quantize"))
        quant_layout.addWidget(QLabel("Levels:"));
        quant_layout.addWidget(self.spin_quant);
        quant_layout.addWidget(btn_rst_quant)
        group_quant.setLayout(quant_layout);
        fund_layout.addWidget(group_quant)

        group_affine = QGroupBox("Affine Transforms")
        aff_layout = QVBoxLayout()
        h_flip = QHBoxLayout()
        h_flip.addWidget(self.create_btn("Flip H", lambda: self.add_op("flip", "Flip H", {"mode": 1})))
        h_flip.addWidget(self.create_btn("Flip V", lambda: self.add_op("flip", "Flip V", {"mode": 0})))
        h_flip.addWidget(self.create_btn("R", lambda: self.reset_op_by_action("flip"), "SmallReset"))
        aff_layout.addLayout(h_flip)

        h_rot = QHBoxLayout()
        self.slider_rot = QSlider(Qt.Horizontal);
        self.slider_rot.setRange(-180, 180);
        self.slider_rot.setValue(0)
        self.lbl_rot = QLabel("0°")
        self.slider_rot.valueChanged.connect(
            lambda v: (self.lbl_rot.setText(f"{v}°"), self.add_op("rotate", f"Rotate {v}°", {"angle": v})))
        btn_rst_rot = QPushButton("R");
        btn_rst_rot.setObjectName("SmallReset")
        btn_rst_rot.clicked.connect(lambda: self.safe_reset_control(self.slider_rot, 0, "rotate"))
        h_rot.addWidget(QLabel("Rot:"));
        h_rot.addWidget(self.slider_rot);
        h_rot.addWidget(self.lbl_rot);
        h_rot.addWidget(btn_rst_rot)
        aff_layout.addLayout(h_rot)

        h_scale = QHBoxLayout()
        self.spin_scale = QDoubleSpinBox();
        self.spin_scale.setRange(0.1, 5.0);
        self.spin_scale.setValue(1.0);
        self.spin_scale.setSingleStep(0.1)
        self.spin_scale.valueChanged.connect(lambda v: self.add_op("scale", f"Scale {v:.1f}x", {"x": v, "y": v}))
        btn_rst_sc = QPushButton("R");
        btn_rst_sc.setObjectName("SmallReset")
        btn_rst_sc.clicked.connect(lambda: self.safe_reset_control(self.spin_scale, 1.0, "scale"))
        h_scale.addWidget(QLabel("Scale:"));
        h_scale.addWidget(self.spin_scale);
        h_scale.addWidget(btn_rst_sc)
        aff_layout.addLayout(h_scale)
        group_affine.setLayout(aff_layout);
        fund_layout.addWidget(group_affine)
        fund_layout.addStretch();
        self.tabs.addTab(tab_fund, "Fundamental")

        # --- TAB 2 ---
        tab_int = QWidget();
        int_layout = QVBoxLayout(tab_int)

        group_int = QGroupBox("Intensity")
        g_layout = QVBoxLayout()
        h_gam = QHBoxLayout()
        self.slider_gamma = QSlider(Qt.Horizontal);
        self.slider_gamma.setRange(1, 30);
        self.slider_gamma.setValue(10)
        self.lbl_gamma = QLabel("1.0")
        self.slider_gamma.valueChanged.connect(lambda v: (self.lbl_gamma.setText(str(v / 10.0)),
                                                          self.add_op("gamma", f"Gamma ({v / 10.0})",
                                                                      {"value": v / 10.0})))
        btn_rst_gam = QPushButton("R");
        btn_rst_gam.setObjectName("SmallReset")
        btn_rst_gam.clicked.connect(lambda: self.safe_reset_control(self.slider_gamma, 10, "gamma"))
        h_gam.addWidget(QLabel("Gamma:"));
        h_gam.addWidget(self.slider_gamma);
        h_gam.addWidget(self.lbl_gamma);
        h_gam.addWidget(btn_rst_gam)
        g_layout.addLayout(h_gam)

        h_con = QHBoxLayout()
        self.slider_cont = QSlider(Qt.Horizontal);
        self.slider_cont.setRange(0, 30);
        self.slider_cont.setValue(10)
        self.lbl_cont = QLabel("1.0")
        self.slider_cont.valueChanged.connect(lambda v: (self.lbl_cont.setText(str(v / 10.0)),
                                                         self.add_op("contrast", f"Contrast ({v / 10.0})",
                                                                     {"value": v / 10.0})))
        btn_rst_con = QPushButton("R");
        btn_rst_con.setObjectName("SmallReset")
        btn_rst_con.clicked.connect(lambda: self.safe_reset_control(self.slider_cont, 10, "contrast"))
        h_con.addWidget(QLabel("Contr:"));
        h_con.addWidget(self.slider_cont);
        h_con.addWidget(self.lbl_cont);
        h_con.addWidget(btn_rst_con)
        g_layout.addLayout(h_con)
        group_int.setLayout(g_layout);
        int_layout.addWidget(group_int)

        group_sp = QGroupBox("Spatial Filters")
        sp_layout = QVBoxLayout()
        self.combo_sp = QComboBox();
        self.combo_sp.addItems(
            ["Mean Blur", "Gaussian Blur", "Median Blur", "Unsharp Masking", "Laplacian", "Sobel", "Prewitt",
             "Roberts"])
        self.combo_sp.currentIndexChanged.connect(self.trigger_spatial_filter)
        h_kern = QHBoxLayout()
        self.spin_k = QSpinBox();
        self.spin_k.setRange(1, 31);
        self.spin_k.setSingleStep(2);
        self.spin_k.setValue(3)
        self.spin_k.valueChanged.connect(self.trigger_spatial_filter)
        btn_rst_sp = QPushButton("Seçiliyi Sıfırla");
        btn_rst_sp.setObjectName("SmallReset")
        btn_rst_sp.clicked.connect(self.reset_current_spatial_filter)
        h_kern.addWidget(QLabel("Kernel:"));
        h_kern.addWidget(self.spin_k)
        sp_layout.addWidget(self.combo_sp);
        sp_layout.addLayout(h_kern);
        sp_layout.addWidget(btn_rst_sp)
        group_sp.setLayout(sp_layout);
        int_layout.addWidget(group_sp)

        h_hist = QHBoxLayout()
        btn_hist_show = QPushButton("Histogram Göster");
        btn_hist_show.clicked.connect(self.show_histogram)
        btn_hist_eq = QPushButton("Hist. Eşitle");
        btn_hist_eq.clicked.connect(lambda: self.add_op("hist_eq", "Hist. Equalization"))
        btn_rst_heq = QPushButton("R");
        btn_rst_heq.setObjectName("SmallReset");
        btn_rst_heq.clicked.connect(lambda: self.reset_op_by_action("hist_eq"))
        h_hist.addWidget(btn_hist_show);
        h_hist.addWidget(btn_hist_eq);
        h_hist.addWidget(btn_rst_heq)
        int_layout.addLayout(h_hist)

        int_layout.addStretch();
        self.tabs.addTab(tab_int, "Intensity/Spatial")

        # --- TAB 3 ---
        tab_morph = QWidget();
        morph_layout = QVBoxLayout(tab_morph)
        group_morph = QGroupBox("Morphological Ops")
        m_layout = QVBoxLayout()
        self.combo_morph = QComboBox();
        self.combo_morph.addItems(
            ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top-Hat", "Black-Hat", "Boundary Extraction",
             "Skeletonization"])
        self.combo_morph.currentIndexChanged.connect(self.trigger_morph_op)
        h_mk = QHBoxLayout()
        self.spin_mk = QSpinBox();
        self.spin_mk.setRange(3, 31);
        self.spin_mk.setSingleStep(2);
        self.spin_mk.setValue(3)
        self.spin_mk.valueChanged.connect(self.trigger_morph_op)
        btn_rst_morph = QPushButton("Seçiliyi Sıfırla");
        btn_rst_morph.setObjectName("SmallReset")
        btn_rst_morph.clicked.connect(self.reset_current_morph_op)
        h_mk.addWidget(QLabel("Kernel:"));
        h_mk.addWidget(self.spin_mk)
        m_layout.addWidget(self.combo_morph);
        m_layout.addLayout(h_mk);
        m_layout.addWidget(btn_rst_morph)
        group_morph.setLayout(m_layout);
        morph_layout.addWidget(group_morph)
        morph_layout.addStretch();
        self.tabs.addTab(tab_morph, "Morphology")

        # --- TAB 4 ---
        tab_seg = QWidget();
        seg_layout = QVBoxLayout(tab_seg)
        group_th = QGroupBox("Thresholding")
        th_layout = QVBoxLayout()
        self.combo_th = QComboBox();
        self.combo_th.addItems(["Global Threshold", "Otsu Threshold", "Adaptive Mean", "Adaptive Gaussian"])
        self.combo_th.currentIndexChanged.connect(self.trigger_threshold_op)
        self.slider_th = QSlider(Qt.Horizontal);
        self.slider_th.setRange(0, 255);
        self.slider_th.setValue(127)
        self.lbl_th_val = QLabel("127")
        self.slider_th.valueChanged.connect(lambda v: (self.lbl_th_val.setText(str(v)), self.trigger_threshold_op()))
        btn_rst_th = QPushButton("Sıfırla");
        btn_rst_th.setObjectName("SmallReset")
        btn_rst_th.clicked.connect(lambda: self.reset_op_by_action("thresh"))
        h_t = QHBoxLayout();
        h_t.addWidget(self.slider_th);
        h_t.addWidget(self.lbl_th_val);
        th_layout.addLayout(h_t)
        th_layout.addWidget(self.combo_th);
        th_layout.addWidget(btn_rst_th);
        group_th.setLayout(th_layout);
        seg_layout.addWidget(group_th)

        group_canny = QGroupBox("Canny Edge")
        c_layout = QVBoxLayout()
        h_c1 = QHBoxLayout();
        h_c1.addWidget(QLabel("T1:"));
        self.spin_c1 = QSpinBox();
        self.spin_c1.setRange(0, 255);
        self.spin_c1.setValue(100);
        h_c1.addWidget(self.spin_c1)
        h_c2 = QHBoxLayout();
        h_c2.addWidget(QLabel("T2:"));
        self.spin_c2 = QSpinBox();
        self.spin_c2.setRange(0, 255);
        self.spin_c2.setValue(200);
        h_c2.addWidget(self.spin_c2)
        self.spin_c1.valueChanged.connect(self.trigger_canny)
        self.spin_c2.valueChanged.connect(self.trigger_canny)
        btn_rst_canny = QPushButton("Sıfırla");
        btn_rst_canny.setObjectName("SmallReset")
        btn_rst_canny.clicked.connect(lambda: self.reset_op_by_action("canny"))
        c_layout.addLayout(h_c1);
        c_layout.addLayout(h_c2);
        c_layout.addWidget(btn_rst_canny)
        group_canny.setLayout(c_layout);
        seg_layout.addWidget(group_canny)

        h_seg_extra = QHBoxLayout()
        btn_hough = self.create_btn("Hough Lines", lambda: self.add_op("hough", "Hough Lines"))
        btn_rst_hough = self.create_btn("R", lambda: self.reset_op_by_action("hough"), "SmallReset")
        btn_comp = self.create_btn("Connected Comp.", lambda: self.add_op("components", "Conn. Components"))
        btn_rst_comp = self.create_btn("R", lambda: self.reset_op_by_action("components"), "SmallReset")
        h_seg_extra.addWidget(btn_hough);
        h_seg_extra.addWidget(btn_rst_hough)
        h_seg_extra.addWidget(btn_comp);
        h_seg_extra.addWidget(btn_rst_comp)
        seg_layout.addLayout(h_seg_extra)
        seg_layout.addStretch();
        self.tabs.addTab(tab_seg, "Segmentation")

        left_layout.addWidget(self.tabs)

        # --- ORTA PANEL ---
        image_scroll = QScrollArea();
        image_scroll.setWidgetResizable(True)
        image_scroll.setStyleSheet("background-color: #111;")
        self.image_label = QLabel("Görüntü Yok");
        self.image_label.setAlignment(Qt.AlignCenter)
        image_scroll.setWidget(self.image_label)

        # --- SAĞ PANEL ---
        right_panel = QWidget();
        right_panel.setFixedWidth(250);
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("<b>Uygulanmış İşlemler</b>"))
        self.stack_list = QListWidget();
        # Seçim modunu NoSelection yapıyoruz, asıl işi checkboxlar yapacak
        self.stack_list.setSelectionMode(QAbstractItemView.NoSelection)
        right_layout.addWidget(self.stack_list)

        # BUTON İSMİ GÜNCELLENDİ
        btn_del_layer = QPushButton("Seçili İşlemleri Sil");
        btn_del_layer.setObjectName("DeleteBtn")
        btn_del_layer.clicked.connect(self.remove_checked_ops)  # Fonksiyon değişti
        right_layout.addWidget(btn_del_layer)

        main_layout.addWidget(left_panel);
        main_layout.addWidget(image_scroll, 1);
        main_layout.addWidget(right_panel)

    # --- YARDIMCILAR ---
    def create_btn(self, text, func, obj_name=None):
        btn = QPushButton(text)
        btn.clicked.connect(func)
        if obj_name: btn.setObjectName(obj_name)
        return btn

    # --- GÜVENLİ SIFIRLAMA ---
    def safe_reset_control(self, widget, default_val, action_name):
        self.is_programmatic_update = True
        widget.blockSignals(True)
        widget.setValue(default_val)
        widget.blockSignals(False)
        self.is_programmatic_update = False
        self.reset_op_by_action(action_name)

    # --- SIFIRLAMA & LİSTE GÜNCELLEME ---
    def reset_op_by_action(self, action_name):
        new_stack = [op for op in self.operation_stack if op['action'] != action_name]
        if len(new_stack) != len(self.operation_stack):
            self.operation_stack = new_stack
            self.update_image_from_stack()
            self.update_stack_list_visuals()

    def reset_current_spatial_filter(self):
        ftype = self.combo_sp.currentText()
        code_map = {"Mean Blur": "blur", "Gaussian Blur": "gaussian", "Median Blur": "median",
                    "Unsharp Masking": "unsharp", "Laplacian": "laplacian", "Sobel": "sobel"}
        target = code_map.get(ftype, "sobel")
        self.reset_op_by_action(target)

    def reset_current_morph_op(self):
        op_name = self.combo_morph.currentText()
        target = op_name.lower().replace("-", "").replace(" ", "")
        if "extraction" in target: target = "boundary"
        if "skeleton" in target: target = "skeleton"

        new_stack = []
        changed = False
        for op in self.operation_stack:
            if op['action'] == 'morph' and op['params'].get('op') == target:
                changed = True
                continue
            new_stack.append(op)

        if changed:
            self.operation_stack = new_stack
            self.update_image_from_stack()
            self.update_stack_list_visuals()

    # --- CHECKBOX İLE SİLME FONKSİYONU ---
    def remove_checked_ops(self):
        """Listede checkbox'ı işaretli olanları siler."""
        indices_to_remove = []

        # Listeyi gez ve işaretli olanların indeksini bul
        for i in range(self.stack_list.count()):
            item = self.stack_list.item(i)
            if item.checkState() == Qt.Checked:
                indices_to_remove.append(i)

        if not indices_to_remove: return

        # Ters sırala (Sondan başa silmek için)
        indices_to_remove.sort(reverse=True)

        # Veri listesinden ve UI listesinden sil
        for index in indices_to_remove:
            if index < len(self.operation_stack):
                del self.operation_stack[index]

        self.update_image_from_stack()
        self.update_stack_list_visuals()  # Listeyi temizleyip yeniden doldurur

    def update_stack_list_visuals(self):
        """Listeyi operation_stack ile senkronize eder ve Checkbox ekler."""
        self.stack_list.clear()
        for op in self.operation_stack:
            item = QListWidgetItem(op["name"])
            # Checkbox özelliğini ekle
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)  # Varsayılan boş
            self.stack_list.addItem(item)

    # --- TRIGGER ---
    def trigger_spatial_filter(self):
        ftype = self.combo_sp.currentText();
        k = self.spin_k.value();
        if k % 2 == 0: k += 1
        code_map = {"Mean Blur": "blur", "Gaussian Blur": "gaussian", "Median Blur": "median",
                    "Unsharp Masking": "unsharp", "Laplacian": "laplacian", "Sobel": "sobel"}
        act = code_map.get(ftype, "sobel")
        self.add_op(act, f"{ftype} (k={k})", {"k": k})

    def trigger_morph_op(self):
        op = self.combo_morph.currentText();
        k = self.spin_mk.value();
        if k % 2 == 0: k += 1
        code = op.lower().replace("-", "").replace(" ", "")
        if "extraction" in code: code = "boundary"
        if "skeleton" in code: code = "skeleton"
        self.add_op("morph", f"{op} (k={k})", {"op": code, "k": k})

    def trigger_threshold_op(self):
        t = self.combo_th.currentText();
        val = self.slider_th.value();
        m = "global"
        if "Otsu" in t:
            m = "otsu"
        elif "Mean" in t:
            m = "adaptive_mean"
        elif "Gaussian" in t:
            m = "adaptive_gauss"
        self.add_op("thresh", f"{t} ({val if m == 'global' else 'Auto'})", {"method": m, "val": val})

    def trigger_canny(self):
        self.add_op("canny", f"Canny ({self.spin_c1.value()},{self.spin_c2.value()})",
                    {"t1": self.spin_c1.value(), "t2": self.spin_c2.value()})

    # --- CORE: STACK YÖNETİMİ ---
    def add_op(self, action, display_name, params=None):
        if self.is_programmatic_update or self.original_image is None: return
        if params is None: params = {}

        # Güncelleme
        if self.operation_stack:
            last_op = self.operation_stack[-1]
            if last_op["action"] == action:
                l_sub = last_op["params"].get("op") or last_op["params"].get("method")
                c_sub = params.get("op") or params.get("method")
                if l_sub == c_sub:
                    last_op["params"] = params
                    last_op["name"] = display_name

                    # Son öğenin ismini güncelle (Checkbox durumunu korumaya çalışmıyoruz çünkü değişiyor)
                    self.update_stack_list_visuals()
                    self.update_image_from_stack()
                    return

        self.operation_stack.append({"action": action, "name": display_name, "params": params})
        self.update_stack_list_visuals()
        self.update_image_from_stack()

    def update_image_from_stack(self):
        """
        Harici ImageEngine kullanarak görüntüyü işler.
        """
        if self.original_image is None: return

        # Harici işlemciyi çağır
        processed_img = ImageEngine.process_stack(self.original_image, self.operation_stack)

        if processed_img is not None:
            self.current_image = processed_img
            self.display_image(self.current_image)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                data = np.fromfile(path, dtype=np.uint8)
                self.original_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.reset_all()
            except:
                QMessageBox.critical(self, "Hata", "Resim okunamadı.")

    def save_image(self):
        if self.current_image is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "sonuc.png", "Images (*.png *.jpg)")
        if path:
            try:
                save_img = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
                ext = os.path.splitext(path)[1];
                _, buf = cv2.imencode(ext, save_img)
                with open(path, "wb") as f:
                    buf.tofile(f)
                QMessageBox.information(self, "Tamam", "Kaydedildi.")
            except:
                QMessageBox.critical(self, "Hata", "Kaydedilemedi.")

    def close_image(self):
        # Resmi ekrandan ve hafızadan siler, her şeyi sıfırlar.
        self.original_image = None
        self.current_image = None
        self.operation_stack = []

        self.stack_list.clear()
        self.image_label.clear()
        self.image_label.setText("Görüntü Yok")

    def reset_all(self):
        self.operation_stack = []
        self.stack_list.clear()
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
        else:
            self.image_label.clear()
            self.image_label.setText("Görüntü Yok")

    def display_image(self, img):
        if img is None: return
        img = np.ascontiguousarray(img)
        if len(img.shape) == 2:
            h, w = img.shape;
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = img.shape;
            q_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.parent().size(), Qt.KeepAspectRatio,
                                                                   Qt.SmoothTransformation))

    def show_histogram(self):
        if self.current_image is None: return
        plt.figure()
        if len(self.current_image.shape) == 3:
            for i, c in enumerate(['r', 'g', 'b']): plt.plot(
                cv2.calcHist([self.current_image], [i], None, [256], [0, 256]), color=c)
        else:
            plt.plot(cv2.calcHist([self.current_image], [0], None, [256], [0, 256]), color='k')
        plt.title("Histogram");
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageProcessorApp()
    win.show()
    sys.exit(app.exec_())