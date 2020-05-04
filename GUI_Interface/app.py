# -*- encoding: utf-8 -*-
'''
@File    :   app.py
@Author  :   Huang Mengqi 
@Version :   1.0
@Contact :   huangmq@mail.ustc.edu.cn
@Last Modified :   2020/04/30 15:55:04
'''

# here put the import lib
import sys
from PyQt5.QtWidgets import QToolTip, QPushButton, QDesktopWidget, QMainWindow, QComboBox
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QLabel, QLineEdit, QGridLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtCore import QFileInfo
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap

from sample import draw_picture 

from sample2 import draw_img
class Example(QWidget):
    
    def __init__(self):
        super().__init__()
        self.label = 'bird'
        self.initUI() #界面绘制交给InitUi方法
        
        
    def initUI(self):
        caption = QLabel('文本描述:')
        caption.setAlignment(QtCore.Qt.AlignCenter)
        captionEdit = QLineEdit()
        captionEdit.setPlaceholderText('请在此处输入英语文本描述...')

        okbtn = QPushButton("确认")
        generate_btn = QPushButton("Let's rock!")
        next_btn = QPushButton("不喜欢？下一张！")

        # 图片
        self.pixmap = QPixmap("images/ustc.jpg")
        self.image = QLabel(self)
        # image.setScaledContents (True)  # 让图片自适应label大小
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setPixmap(self.pixmap)

        # 此处加入选择模型的下拉列表
        combo = QComboBox(self)
        combo.addItem("鸟类图片生成模型")
        combo.addItem("通用图片生成模型")
        combo.addItem("通用图片生成模型2")



        grid = QGridLayout()
        grid.setSpacing(10)  # 表示各个控件之间的上下间距

        grid.addWidget(caption, 1, 0)
        grid.addWidget(captionEdit, 1, 1)
        grid.addWidget(okbtn, 1, 2)

        grid.addWidget(generate_btn, 3, 2)
        grid.addWidget(next_btn, 4, 2)
        grid.addWidget(combo, 2, 2)
        grid.addWidget(self.image, 2, 0, 3, 2)

        self.setLayout(grid) 

        okbtn.clicked.connect(lambda : self.set_text(captionEdit.text()))
        generate_btn.clicked.connect(lambda : self.draw_image())
        next_btn.clicked.connect(lambda : self.draw_image())
        combo.activated[str].connect(self.onActivated)

        #这种静态的方法设置一个用于显示工具提示的字体。我们使用10px滑体字体
        QToolTip.setFont(QFont('SansSerif', 10))
        #创建一个提示，我们称之为settooltip()方法。我们可以使用丰富的文本格式
        self.setToolTip('This is HMQ\'s graduation project!')


        #设置窗口的位置和大小
        self.resize(800, 500)
        self.center()

        #设置窗口的标题
        self.setWindowTitle('Draw what you write!')
        #设置窗口的图标，引用当前目录下的web.png图片
        root = QFileInfo(__file__).absolutePath()
        self.setWindowIcon(QIcon(root+'/images/学士帽.png'))        
        
        #显示窗口
        self.show()
    
    def onActivated(self, text):
        if text == '鸟类图片生成模型':
            self.label = 'bird'
        elif text == '通用图片生成模型':
            self.label = 'coco'
        elif text == '通用图片生成模型2':
            self.label = 'coco2'
    
    def draw_image(self):
        if self.label == 'bird' or self.label == 'coco':
            print(self.label)
            draw_picture(self.label)
        else:
            print(self.label)
            draw_img()
        self.show_image()
    
    def show_image(self):
        # self.image.setPixmap(QPixmap(""))   # 移除label上的图片
        self.pixmap = QPixmap("results/0_0_g2.png")
        self.image.setPixmap(self.pixmap)

    
    def set_text(self, text):
        with open('caption.txt', 'w') as f:
            f.write(text)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                "确认退出吗？", QMessageBox.Yes |
                QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
    def center(self):
        #获得窗口
        qr = self.frameGeometry()
        #获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        #显示到屏幕中心
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
if __name__ == '__main__':
    #创建应用程序和对象
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_()) 