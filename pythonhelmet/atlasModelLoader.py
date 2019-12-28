import hiai
from hiai.nn_tensor_lib import DataType
import flask_jpegHandler
import os
import numpy as np
import time
import cv2 as cv
from flask import Flask,request,redirect,url_for,jsonify
import json

name_index_map = {1:'red',2:'white',3:'yellow',4:'blue',5:'none'}

def CreateGraph(model, modelInWidth, modelInHeight, dvppInWidth, dvppInHeight):
    myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()
    if myGraph is None:
        print 'get defaule graph failed'
        return None
    print 'dvppwidth %d, dvppheight %d' % (dvppInWidth, dvppInHeight)
    cropConfig = hiai.CropConfig(0, 0, dvppInWidth, dvppInHeight)
    print 'cropConfig ', cropConfig
    resizeConfig = hiai.ResizeConfig(modelInWidth, modelInHeight)
    print 'resizeConfig ', resizeConfig

    nntensorList = hiai.NNTensorList()
    print 'nntensorList', nntensorList

    resultCrop = hiai.crop(nntensorList, cropConfig)
    print 'resultCrop', resultCrop

    resultResize = hiai.resize(resultCrop, resizeConfig)
    print 'resultResize', resultResize

    resultInference = hiai.inference(resultResize, model, None)
    print 'resultInference', resultInference

    if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
        print 'create graph ok !!!!'
        return myGraph
    else:
        print 'create graph failed, please check Davinc log.'
        return None


def CreateGraphWithoutDVPP(model):
    print model
    myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()
    print myGraph
    if myGraph is None:
        print 'get defaule graph failed'
        return None

    nntensorList = hiai.NNTensorList()
    print nntensorList

    resultInference = hiai.inference(nntensorList, model, None)
    print nntensorList
    print hiai.HiaiPythonStatust.HIAI_PYTHON_OK
    # print myGraph.create_graph()

    if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
        print 'create graph ok !!!!'
        return myGraph
    else:
        print 'create graph failed, please check Davinc log.'
        return None


def GraphInference(graphHandle, inputTensorList):
    if not isinstance(graphHandle, hiai.Graph):
        print "graphHandle is not Graph object"
        return None

    resultList = graphHandle.proc(inputTensorList)

    return resultList


def SSDPostProcess_returnbbox(resultList):
    if resultList is not None:
        instance_num = int(resultList[1][0][0][0][0])
        tensor_bbox = resultList[0]

        res = {
            'results': []
        }
        for anchor in range(instance_num):
            class_idx = tensor_bbox[anchor][0][0][1]
            lt_x = tensor_bbox[anchor][0][0][3]
            lt_y = tensor_bbox[anchor][0][0][4]
            rb_x = tensor_bbox[anchor][0][0][5]
            rb_y = tensor_bbox[anchor][0][0][6]
            score = tensor_bbox[anchor][0][0][2]
            res['results'].append({
                'location': {
                    'left': float(lt_x),
                    'top': float(lt_y),
                    'width': float(abs(rb_x - lt_x)),
                    'height': float(abs(rb_y - lt_y))},
                'name': str(name_index_map[int(class_idx)]),
                'score': float(score),
            })

        if len(res) == 0:
            return None
        else:
            return res

    else:
        print 'graph inference failed '
        return None

if __name__ == "__main__":
    print("This is a atlas model loader lib")