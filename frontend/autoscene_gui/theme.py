"""AutoScene desktop theme: restrained engineering UI with clear action hierarchy.

v2 changes:
  - Font size hierarchy: 13px GroupBox titles, 12px body, 11px secondary
  - State color semantics: idle=gray, preparing=blue, running=green, error=red
  - Loading/disabled states with visual feedback
  - Improved scrollbar contrast and table zebra striping
  - Status indicator badge styles
"""

APP_STYLE = r"""
/* ==================================================================
   Global
   ================================================================== */
QMainWindow, QWidget {
    background: #f4f6f8;
    color: #24303f;
    font-family: "PingFang SC", "SF Pro Text", "Microsoft YaHei";
    font-size: 12px;
}

/* ==================================================================
   Toolbar
   ================================================================== */
QToolBar#mainToolbar {
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #dfe5eb;
    spacing: 6px;
    padding: 7px 10px;
}
QToolBar#mainToolbar::separator {
    background: #dfe5eb;
    width: 1px;
    margin: 5px 7px;
}
QToolBar QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 9px;
    color: #516071;
}
QToolBar QToolButton:hover {
    background: #f0f4f8;
    border-color: #d8e0e8;
}

/* ==================================================================
   Buttons — 层级体系
   ================================================================== */
QPushButton {
    min-height: 28px;
    padding: 2px 12px;
    border: 1px solid #cdd6df;
    border-radius: 6px;
    background: #ffffff;
    color: #334155;
}
QPushButton:hover {
    background: #f7fafc;
    border-color: #8ba0b5;
}
QPushButton:pressed {
    background: #e8eef4;
    border-color: #71869b;
}
QPushButton:disabled {
    color: #a9b2bc;
    background: #eef1f4;
    border-color: #e1e5e9;
}

/* loading state — 异步操作进行中 */
QPushButton[loading="true"] {
    color: #2563eb;
    background: #eff6ff;
    border-color: #93c5fd;
    font-style: italic;
}

QPushButton#primaryButton {
    color: #ffffff;
    background: #2563eb;
    border-color: #2563eb;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background: #1d4ed8; border-color: #1d4ed8; }
QPushButton#primaryButton:disabled {
    background: #93b8f5;
    border-color: #93b8f5;
    color: #dbeafe;
}

QPushButton#prepareButton {
    color: #1e40af;
    background: #eff6ff;
    border-color: #93c5fd;
    font-weight: 600;
}
QPushButton#loadButton { font-weight: 600; }

QPushButton#dangerButton {
    color: #b42318;
    background: #fff5f4;
    border-color: #f3b8b3;
}

QPushButton#placeButton:checked {
    color: #ffffff;
    background: #0f766e;
    border-color: #0f766e;
}

QPushButton#mapToolButton {
    min-height: 24px;
    padding: 1px 9px;
    border-color: #d5dde5;
    background: #f8fafc;
}

/* ==================================================================
   Map toolbar
   ================================================================== */
QWidget#mapToolbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}
QLabel#mapHint { color: #718096; font-size: 11px; }
QLabel#zoomReadout {
    color: #475569;
    background: #eef2f6;
    border-radius: 5px;
    padding: 4px 7px;
}
QGraphicsView#sceneView {
    background: #f8fafb;
    border: none;
}

/* ==================================================================
   GroupBox — 分区容器
   ================================================================== */
QGroupBox {
    background: #ffffff;
    border: 1px solid #dfe5eb;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
    font-size: 13px;
    color: #334155;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #475569;
    font-size: 12px;
}

/* ==================================================================
   Form controls
   ================================================================== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 27px;
    background: #ffffff;
    border: 1px solid #cfd8e3;
    border-radius: 5px;
    padding: 1px 7px;
    selection-background-color: #bfdbfe;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #3b82f6;
}
QComboBox::drop-down { border: none; width: 22px; }

/* ==================================================================
   Tab widget
   ================================================================== */
QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #dfe5eb;
    border-radius: 7px;
    top: -1px;
}
QTabBar::tab {
    background: #e9eef3;
    color: #637083;
    border: 1px solid #d9e0e7;
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    color: #1d4ed8;
    background: #ffffff;
    font-weight: 600;
}
QTabBar::tab:hover:!selected { background: #f1f5f9; color: #334155; }

/* ==================================================================
   Tables & Lists
   ================================================================== */
QTableWidget, QListWidget, QPlainTextEdit {
    background: #ffffff;
    alternate-background-color: #f7f9fb;
    border: 1px solid #dfe5eb;
    border-radius: 6px;
    gridline-color: #edf1f5;
    selection-background-color: #dbeafe;
    selection-color: #1e3a5f;
}
QTableWidget::item { padding: 5px; }
QHeaderView::section {
    background: #e8edf3;
    color: #45556b;
    border: none;
    border-right: 1px solid #dbe2e9;
    border-bottom: 1px solid #dbe2e9;
    padding: 7px 6px;
    font-weight: 600;
    font-size: 11px;
}

/* 表格数值列加粗，提升可读性 */
QTableWidget::item[role="metric"] {
    font-weight: 600;
}

/* ==================================================================
   Checkbox
   ================================================================== */
QCheckBox { spacing: 6px; color: #526173; }
QCheckBox::indicator { width: 15px; height: 15px; }

/* ==================================================================
   Splitter
   ================================================================== */
QSplitter::handle { background: #e2e8f0; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }

/* ==================================================================
   Status bar
   ================================================================== */
QStatusBar {
    background: #ffffff;
    color: #657386;
    border-top: 1px solid #dfe5eb;
    font-size: 11px;
}

/* ==================================================================
   Scrollbar — 更明显的拖拽手柄
   ================================================================== */
QScrollBar:vertical {
    background: #f4f6f8;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #bcc5d0;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #9aa5b3; }
QScrollBar:horizontal {
    background: #f4f6f8;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #bcc5d0;
    border-radius: 4px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: #9aa5b3; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }

/* ==================================================================
   State badges — 运行状态彩色标签
   ================================================================== */
QLabel#modeBadge {
    border-radius: 6px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}
QLabel#dataBadge {
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
}
QLabel#linkBadge {
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* state 文字颜色——由代码动态设置 styleSheet */
QLabel#stateLabel { font-weight: 600; font-size: 13px; }

/* ==================================================================
   Dark editor panels — 嵌入式暗色面板
   ================================================================== */
QPlainTextEdit[dark="true"] {
    background: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #4a5568;
    border-radius: 6px;
    selection-background-color: #264f78;
}

/* ==================================================================
   Modal / overlay
   ================================================================== */
QMessageBox {
    background: #ffffff;
}
QMessageBox QLabel {
    color: #24303f;
    font-size: 12px;
}
"""  # end stylesheet
