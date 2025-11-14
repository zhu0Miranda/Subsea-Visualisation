from PyQt5.QtGui import QValidator, QIntValidator, QIcon
from PyQt5.QtWidgets import QApplication, QLineEdit, QToolBar, QStatusBar, QLabel
from PyQt5 import uic
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
# 工具箱
    ui = uic.loadUi(r"F:\海底探测\project1\菜单设置.ui")
    myToolBar_1: QToolBar = ui.toolBar
    myToolBar_1.addAction(QIcon(r"F:\海底探测\图标\add-document.png"), "新建工程")
    myToolBar_2: QToolBar = ui.toolBar
    myToolBar_2.addAction(QIcon(r"F:\海底探测\图标\folder.png"), "打开工程")
    myToolBar_3: QToolBar = ui.toolBar
    myToolBar_3.addAction(QIcon(r"F:\海底探测\图标\document.png"), "保存工程")
    myToolBar_4: QToolBar = ui.toolBar
    myToolBar_4.addAction(QIcon(r"F:\海底探测\图标\Closure of works.png"), "关闭工程")
    myToolBar_5: QToolBar = ui.toolBar
    myToolBar_5.addAction(QIcon(r"F:\海底探测\图标\grout.png"), "接口设置")
    myToolBar_6: QToolBar = ui.toolBar
    myToolBar_6.addAction(QIcon(r"F:\海底探测\图标\setting.png"), "参数设置")
    myToolBar_7: QToolBar = ui.toolBar
    myToolBar_7.addAction(QIcon(r"F:\海底探测\图标\play.png"), "开始工作")
    myToolBar_8: QToolBar = ui.toolBar
    myToolBar_8.addAction(QIcon(r"F:\海底探测\图标\Image Settings.png"), "图像设置")
    myToolBar_9: QToolBar = ui.toolBar
    myToolBar_9.addAction(QIcon(r"F:\海底探测\图标\2D mode.png"), "2D模式")
    myToolBar_10: QToolBar = ui.toolBar
    myToolBar_10.addAction(QIcon(r"F:\海底探测\图标\3D mode.png"), "3D模式")
    myToolBar_11: QToolBar = ui.toolBar
    myToolBar_11.addAction(QIcon(r"F:\海底探测\图标\shuffle.png"), "数据转换")
    myToolBar_12: QToolBar = ui.toolBar
    myToolBar_12.addAction(QIcon(r"F:\海底探测\图标\rotate-right.png"), "回放模式")

    #状态栏
    myStatusBar: QStatusBar = ui.statusbar

    myLabel = QLabel()
    myLabel.setText("朱朱朱")
    myStatusBar.addWidget(myLabel)
    # myStatusBar.showMessage("当前用户登录中",3000) 临时显示

    ui.show()

    sys.exit(app.exec())