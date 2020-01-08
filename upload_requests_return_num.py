import requests
import json
import cv2
import base64


def convertJPG_to_string(img_path,ask_url,scale_size=[300,300]):
    """

    :param img_path:the image wanted to inference
    :param ask_url: the url in Atlas200DK
    :param scale_size: the model require shape ,the helmetvggssd is [300,300]
    :return: None
    """
    cv_img = cv2.imread(img_path)

    # cv_img = cv2.resize(cv_img,(scale_size[0],scale_size[1]))
    img_encoded = cv2.imencode('.jpg',cv_img)[1].tostring()
    res = requests.post(ask_url, data=img_encoded)
    print(res.text)

if __name__ == '__main__':
    convertJPG_to_string('/home/wuchenxi/PycharmProjects/Atlas200DK/28.jpg',
                         ask_url='http://192.168.8.151:120')
