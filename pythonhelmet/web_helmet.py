#coding=utf-8

import hiai
from atlasModelLoader import *
from hiai.nn_tensor_lib import DataType
import flask_jpegHandler
import os
import numpy as np
import time
import cv2 as cv
from flask import Flask,request,redirect,url_for,jsonify
import json


inferenceModel = None
myGraph = None


def Init(model_path):
	global inferenceModel
	global myGraph
	inferenceModel = hiai.AIModelDescription('helmet_vgg_deploy', model_path)

	# we will resize the jpeg to 896*608 to meet faster-rcnn requirement via opencv,
	# so DVPP resizing is not needed
	myGraph = CreateGraphWithoutDVPP(inferenceModel)
	if myGraph is None:
		print "CreateGraph failed"
		return None


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	try:
		global myGraph
		start = time.time()
		val_image = cv.imdecode(np.frombuffer(request.data, np.uint8), cv.IMREAD_COLOR)
		dvppInWidth = 300
		dvppInHeight = 300
		input_image = flask_jpegHandler.decode2yuv(val_image)

		inputImageTensor = hiai.NNTensor(input_image, dvppInWidth, dvppInHeight, 3, 'testImage', DataType.UINT8_T,
										 dvppInWidth * dvppInHeight * 3 / 2)

		nntensorList = hiai.NNTensorList(inputImageTensor)

		resultList = GraphInference(myGraph, nntensorList)

		res = SSDPostProcess_returnbbox(resultList)

		end = time.time()
		cost_time = 'cost time ' + str((end - start) * 1000) + 'ms'

		return json.dumps([res,cost_time])


	except Exception as err:
		app.logger.error(err)
		return str(err), 403
	

if __name__ == "__main__":
	Init(model_path = '/home/HwHiAiUser/HIAI_PROJECTS/ascend_workspace/pythonhelmet/models/helmet_vgg_300.om')
	app.run(host='192.168.1.2', port=118, debug=False)

