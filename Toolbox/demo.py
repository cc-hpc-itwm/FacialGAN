import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

# OUR
import copy
import kornia
from munch import Munch

from os.path import join as ospj

import torch
import torch.nn.functional as F

from core.model import Generator, MappingNetwork, StyleEncoder
from core.checkpoint import CheckpointIO

from ui import ui
from ui.mouse_event import GraphicsScene


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0),  QColor(51, 51, 255)]

class Ex(QWidget, ui.Ui_Form):
    def __init__(self):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()

        self.output_img = None
        self.mat_img = None
        self.attribute = []
        self.tmp_attribute = None
        self.mask_size = [1.0, 1.0, 1.0]
        self.mask_name = ""
        self.load = False
        self.y_trg = 0
        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None

        self.mouse_clicked = False
        self.scene = QGraphicsScene()
        self.graphicsView_1.setScene(self.scene)
        self.graphicsView_1.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.mask_scene = GraphicsScene(self.mode, self.size)
        self.graphicsView_2.setScene(self.mask_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.ref_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_4.setScene(self.result_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.color = None

        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.pushButton_10.setEnabled(False)
        self.pushButton_11.setEnabled(False)
        self.slidern_1.setEnabled(False)
        self.slidern_2.setEnabled(False)
        self.slidern_3.setEnabled(False)

        #self.mask_scene.size = 6

    def slider(self, value):
        # mapping
        mask_size = 0.85 + 0.003*value
        self.mask_size[self.tmp_attribute] = mask_size
        self.load_mask()

    def open(self):
        self.load = True
        #loading real image
        directory = os.path.join(QDir.currentPath(),"samples/faces")
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", directory)

        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView_1.size(), Qt.IgnoreAspectRatio)

            if len(self.scene.items())>0:
                self.scene.removeItem(self.scene.items()[-1])
            self.scene.addPixmap(image)

        self.mask_name = fileName
        self.load_mask()

    def load_mask(self):
        # loading mask from real image
        skin_mask = np.zeros([256,256])
        nose_mask = np.zeros([256,256])
        eye_mask = np.zeros([256,256])
        mouth_mask = np.zeros([256,256])


        tmp = self.mask_name.split("/")[-1]
        tmp_filename = tmp.split(".")[0].zfill(5)

        directory = os.path.join(QDir.currentPath(),"samples/masks")

        for filename in os.listdir(directory):
            if filename.split("_")[0] == tmp_filename:
                path = os.path.join(directory, filename)
                mat_img = cv2.imread(path,0)
                mat_img = cv2.resize(mat_img, (256,256))
                mat_img = np.round(mat_img/255)
                tmp_split = filename.split("_")[-1]
                tmp_split = tmp_split.split(".")[0]

                if tmp_split == "brow"  or tmp_split == "eye":
                    eye_mask = eye_mask + 1*mat_img
                    eye_mask[eye_mask!=0]=1
                elif tmp_split == "lip"  or tmp_split == "mouth":
                    mouth_mask = mouth_mask + 2*mat_img
                    mouth_mask[mouth_mask!=0]=2
                elif tmp_split == "nose":
                    nose_mask = 3*mat_img
                    nose_mask[nose_mask!=0]=3
                elif tmp_split == "skin":
                    skin_mask =  4*mat_img
                    skin_mask[skin_mask!=0]=4

            else:
                continue

        # resikze mask
        eye_mask = torch.tensor(eye_mask)
        size_tmp = int(self.mask_size[0]* eye_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(eye_mask,0),0), size=size_tmp, mode='bilinear')
        eye_mask = kornia.geometry.transform.center_crop(mask_tmp, (256,256))[0,0]
        eye_mask = eye_mask.numpy()
        eye_mask[eye_mask!=0]=1

        mouth_mask = torch.tensor(mouth_mask)
        size_tmp = int(self.mask_size[1]* mouth_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(mouth_mask,0),0), size=size_tmp, mode='bilinear')
        mouth_mask = kornia.geometry.transform.center_crop(mask_tmp, (256,256))[0,0]
        mouth_mask = mouth_mask.numpy()
        mouth_mask[mouth_mask!=0]=2

        nose_mask = torch.tensor(nose_mask)
        size_tmp = int(self.mask_size[2]* nose_mask.size(0))
        mask_tmp = F.interpolate(torch.unsqueeze(torch.unsqueeze(nose_mask,0),0), size=size_tmp, mode='bilinear')
        nose_mask = kornia.geometry.transform.center_crop(mask_tmp, (256,256))[0,0]
        nose_mask = nose_mask.numpy()
        nose_mask[nose_mask!=0]=3

        tmp_mask = skin_mask + eye_mask
        tmp_mask[tmp_mask>4]=1
        tmp_mask = tmp_mask + mouth_mask
        tmp_mask[tmp_mask>5]=2
        tmp_mask = tmp_mask + nose_mask
        tmp_mask[tmp_mask>4]=3

        res_mask = np.repeat(tmp_mask[:, :, np.newaxis], 3, axis=2)
        res_mask = np.asarray(res_mask, dtype=np.uint8)

        self.mask = res_mask.copy()
        self.mask_m = res_mask
        mat_img = res_mask.copy()

        image = QImage(mat_img.data, 256, 256, QImage.Format_RGB888)


        for i in range(256):
            for j in range(256):
                r, g, b, a = image.pixelColor(i, j).getRgb()
                image.setPixel(i, j, color_list[r].rgb())


        pixmap = QPixmap()
        pixmap.convertFromImage(image)
        self.image = pixmap.scaled(self.graphicsView_1.size(), Qt.IgnoreAspectRatio)
        self.mask_scene.reset()
        if len(self.mask_scene.items())>0:
            self.mask_scene.reset_items()
        self.mask_scene.addPixmap(self.image)

    def open_ref(self):
        if self.load == True:
            self.pushButton_4.setEnabled(True)
            self.pushButton_5.setEnabled(True)
            self.pushButton_7.setEnabled(True)
            self.pushButton_8.setEnabled(True)
            self.pushButton_9.setEnabled(True)
            self.pushButton_10.setEnabled(True)
            self.pushButton_11.setEnabled(True)

        # loading random reference image for style
        directory = os.path.join(QDir.currentPath(),"samples/faces")
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", directory)
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.ref = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer", "Cannot load %s." % fileName)
                return

            image = image.scaled(self.graphicsView_1.size(), Qt.IgnoreAspectRatio)
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)

    def apply(self):
        self.pushButton_3.setEnabled(True)

        # updates the changes of the mask
        for i in range(5):
            self.mask_m = self.make_mask(self.mask_m, self.mask_scene.mask_points[i], self.mask_scene.size_points[i], i)


        transform_mask = transforms.Compose([transforms.Resize([256, 256]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0, 0, 0),(1 / 255., 1 / 255., 1 / 255.))
                                        ])

        transform_image = transforms.Compose([transforms.Resize([256, 256]),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                        ])
        mask = self.mask.copy()
        mask_m = self.mask_m.copy()
        img = self.img.copy()
        ref = self.ref.copy()


        mask = transform_mask(Image.fromarray(np.uint8(mask)))
        mask_m = transform_mask(Image.fromarray(np.uint8(mask_m)))
        img = transform_image(img)
        ref = transform_image(ref)

        start_t = time.time()

        s_trg = style_encoder(torch.FloatTensor([ref.numpy()]), torch.LongTensor([self.y_trg]))
        masks = (torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]))
        generated = generator(torch.FloatTensor([img.numpy()]), s_trg, masks=masks, attribute=self.attribute)


        end_t = time.time()
        print('inference time : {}'.format(end_t-start_t))
        result = generated.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()
        result = (result + 1) /2
        result = result.clip(0, 1)
        result = result * 255

        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        result = result.copy()
        self.output_img = result


        qim = QImage(result.data, 256, 256, QImage.Format_RGB888)

        if len(self.result_scene.items())>0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))


    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File", QDir.currentPath())
            try:
                im = Image.fromarray(self.output_img)
                im.save(fileName+'.jpg')
            except:
                pass


    def clear(self):
        self.pushButton_6.setEnabled(False)
        self.slidern_1.setEnabled(False)
        self.slidern_1.setValue(49)
        self.slidern_2.setEnabled(False)
        self.slidern_2.setValue(49)
        self.slidern_3.setEnabled(False)
        self.slidern_3.setValue(49)
        self.attribute = []
        self.mask_size = [1.0, 1.0, 1.0]
        self.tmp_attribute = None
        self.load_mask()


    def man_mode(self):
        self.y_trg = 1

    def woman_mode(self):
        self.y_trg = 0

    def eyes_mode(self):
        self.slidern_1.setEnabled(False)
        self.slidern_2.setEnabled(True)
        self.slidern_3.setEnabled(False)
        self.pushButton_6.setEnabled(True)
        self.mask_scene.mode = 1
        self.tmp_attribute = 0
        if (0 in self.attribute) == False:
            self.attribute.append(0)

    def mouth_mode(self):
        self.slidern_1.setEnabled(False)
        self.slidern_2.setEnabled(False)
        self.slidern_3.setEnabled(True)
        self.pushButton_6.setEnabled(True)
        self.mask_scene.mode = 2
        self.tmp_attribute = 1
        if (1 in self.attribute) == False:
            self.attribute.append(1)

    def nose_mode(self):
        self.slidern_1.setEnabled(True)
        self.slidern_2.setEnabled(False)
        self.slidern_3.setEnabled(False)
        self.pushButton_6.setEnabled(True)
        self.mask_scene.mode = 3
        self.tmp_attribute = 2
        if (2 in self.attribute) == False:
            self.attribute.append(2)

    def skin_mode(self):
        self.mask_scene.mode = 4



def _load_checkpoint(nets_ema, checkpoint_dir, step):
    ckptios = [CheckpointIO(ospj(checkpoint_dir, 'facial_checkpoint.ckpt'), **nets_ema)]
    for ckptio in ckptios:
        ckptio.load(step)

if __name__ == '__main__':

    # hyper-parametrs
    checkpoint_dir = "checkpoints"
    resume_iter = 200000

    # initilize networks
    generator = Generator()
    mapping_network = MappingNetwork()
    style_encoder = StyleEncoder()
    nets_ema = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder)
    # load weights
    _load_checkpoint(nets_ema, checkpoint_dir, resume_iter)

    app = QApplication(sys.argv)
    ex = Ex()
    sys.exit(app.exec_())
