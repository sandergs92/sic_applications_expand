import queue
import cv2

from sic_framework.core.message_python2 import CompressedImageMessage
from sic_framework.devices.common_naoqi.naoqi_camera import TopNaoCamera, BottomNaoCamera
from sic_framework.tests.services.merge_images_service import MergeImages, MergeImageConf

imgs = queue.Queue()
def on_image(image_message: CompressedImageMessage):
    imgs.put(image_message.image)


camera1 = TopNaoCamera(ip="192.168.0.180")
camera2 = BottomNaoCamera(ip="192.168.0.180")


conf = MergeImageConf(TopNaoCamera, BottomNaoCamera)
merger = MergeImages(conf=conf)

merger.connect(camera1)
merger.connect(camera2)

merger.register_callback(on_image)

while True:
    img = imgs.get()
    cv2.imshow('', img)
    cv2.waitKey(1)
