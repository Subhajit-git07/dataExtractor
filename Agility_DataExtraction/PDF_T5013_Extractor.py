#!/usr/bin/env python
# coding: utf-8

import fitz
import numpy as np
import cv2 as cv
import os
import pytesseract
import cv2
import time
import pickle
import re
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from detectron2.engine import DefaultPredictor
from collections import OrderedDict
import easyocr
import matplotlib.pyplot as plt

cfg_save_path = "/pas-models/T5013-Model/t5013_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = "/pas-models/T5013-Model/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

reader = easyocr.Reader(["en"], gpu=False)


class PDF_T5013_ImageExtractor:
    """
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path, template_json):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = 0.9
        self.page_number = 1
        self.template_json = template_json
        
    def convert_pdf_to_image(self):
        try:
            pdf_document = fitz.open(self.pdf_file_path)
            dpi = 300
            first_page = pdf_document.load_page(self.page_number - 1)
            pix = first_page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            pix.save(self.output_image_path)
            pdf_document.close()
            print(f'Conversion of the first page to {self.output_image_path} completed.')
            return True
        except Exception as e:
            print(f"Error: Pdf to Image Conversion failed!! {str(e)}")
            return False
            
    def order_box8(self,output_json):
        kk='Box-Case::Code::Amount - '
        y_list=[]
        temp_d={}
        for i in range(1,13):
            key=kk+str(i)
            y=output_json[key]['Coordinates'][1]
            s=output_json[key]['Coordinates'][0]+y
            y_list.append(y)
            temp_d[key]={"Sum":s,'y':y}
        y_list.sort()
        
        j=1
        kk='Box-Case::Code::Amount - '
        for i in range(0,12,2):
            k1=''
            k2=''
            for k in temp_d.keys():
                if y_list[i] == temp_d[k]['y'] and k1=='':
                    k1=k
                    #print(k1)
                elif y_list[i+1] == temp_d[k]['y']:
                    k2=k
                    #print(k2)
            if temp_d[k1]['Sum'] < temp_d[k2]['Sum']:
                temp_d[k1].update({'Box':kk+str(j)})
                temp_d[k2].update({'Box':kk+str(j+6)})
            else:
                temp_d[k1].update({'Box':kk+str(j+6)})
                temp_d[k2].update({'Box':kk+str(j)})
            j=j+1
        
        new_dic={}
        for key in temp_d.keys():
            new_dic[key]=output_json[key]
            del output_json[key]
        for key in temp_d.keys():
            output_json[temp_d[key]['Box']]=new_dic[key]
            #print(temp_d[key]['Box'],":", new_dic[key])
        return output_json

    def map_box9(self,result): 
        box_9={}
        for i in result:
            isGot=False
            for k in box_9.keys():
                if abs(i[0][0][1]-k) < 10:
                    box_9[k]=box_9[k]+" :: "+i[1]
                    isGot=True
            if not isGot:
                box_9[i[0][0][1]]=i[1]
        
        k_list=list(box_9.keys())
        k_list.sort()
        key='Box-Case :: Code :: Other Information'
        for i in range(len(k_list)):
            if i==0:
                key='Box-Case :: Code :: Other Information'
                del box_9[k_list[i]]
            else:
                box_9[key+" - "+str(i)]=box_9[k_list[i]]
                del box_9[k_list[i]]
        #print(box_9)
        if len(box_9) < 4:
            for i in range(len(box_9)+1,4+1):
                box_9[key+" - "+str(i)]=""
                
        return box_9
        
    def extract_identified_text(self, predictor):
        im = cv2.imread(self.output_image_path)
        outputs = predictor(im)
        boxes_list = outputs["instances"].pred_boxes
        scores_list = outputs["instances"].scores.detach().cpu().numpy()
        pred_classes_list = outputs["instances"].pred_classes.detach().cpu().numpy()
        # print(type(boxes_list))
        # print(type(scores_list))
        #print(pred_classes_list)
        # for b in boxes_list:
        #     print(b)
            
        #box_array = np.array(boxes_list)

        class_wise_coord_dict = {}
        class_8_list=[]
        cnt=793
        for box_cord, score_val, pred_cls_val in zip(boxes_list, scores_list, pred_classes_list):
            bbox = box_cord.detach().cpu().numpy()
            if class_wise_coord_dict.get(pred_cls_val) and pred_cls_val != 8:
                if float(score_val) >  class_wise_coord_dict[pred_cls_val]["score"]:
                    class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score":float(score_val)}
            elif pred_cls_val == 8:
                #class_8_list.append({"bbox": bbox, "score":float(score_val)})
                class_wise_coord_dict[cnt+pred_cls_val] = {"bbox": bbox, "score":float(score_val)}
                cnt=cnt+1
            else:
                #print(pred_cls_val, bbox)
                class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score":float(score_val)}

        pred_cls_data_dict = {}
        for pred_cls, bbox_scores_dict in class_wise_coord_dict.items():
            x_min, y_min, x_max, y_max = list(map(int, bbox_scores_dict["bbox"]))
            coords = [x_min, y_min, x_max, y_max]
            if pred_cls == 15:
                y_min=y_min+int((y_min*20/100))
            #print("pred_cls",pred_cls)
            cropped_image = im[y_min:y_max+20, x_min:x_max+20, :]
            # cropped_image = cv2.resize(cropped_image, (x_max+10, y_max+10))
            # cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)
            grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            _, cropped_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            result = reader.readtext(cropped_image)

            #print(result)
            # if pred_cls == 12:
            #     plt.figure(figsize=(5, 5))
            #     plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            #     plt.show()
            box9Map={}
            if pred_cls == 9:
                #print(pred_cls, result)
                box9Map=self.map_box9(result)
                #print(box9Map)
                pred_cls_data_dict[pred_cls] = (box9Map, coords)
                #print(pred_cls_data_dict[pred_cls])
            # if pred_cls == 8:
            #     box9Map=self.map_box8(result)
            #    # print(box9Map)
            #     pred_cls_data_dict[pred_cls] = (box9Map, coords)
                
            para_res = ''
            if pred_cls != 9:
                for detection in result:
                    para_res = para_res+detection[1]+' '
                #print(pred_cls, para_res)
                pred_cls_data_dict[pred_cls] = (para_res.strip(), coords)

        class_info_dict = {}
        class_info_dict = pred_cls_data_dict
        #print(class_info_dict)
        return class_info_dict

    def mapping_data(self, class_info_dict):
        final_info_dict = {}
        for box_cls, key in self.template_json.items():
            box_cls = int(box_cls)
            if class_info_dict.get(box_cls):
                value, coords =  class_info_dict.get(box_cls)
                final_value = ""
#                 final_info_dict[key] = {"Value":value, "Coordinates":coords}
                #print(box_cls, value)
                if box_cls in (0,1,2,3,4,5,6,7,13,14,16):
                    if box_cls in (4,5,6,7): #and re.search(r' ([+-]?(\d*[,])*\d*[.]?\d+$)',value):
                #print(box_cls, value)
                        #print(box_cls, value)
                        pattern = r" 0\d+0"
                        final_value = re.split(pattern, value)[-1].strip()
                       # exval = re.search(r' ([+-]?(\d*[,])*\d*[.]?\d+$)',value).groups()
                        #final_value = exval[0] if exval[0] else ''
                    elif box_cls in (0,1,2,3,13,14): #re.search(r'.* (\d+)',value):
                        # Define a regex pattern to split the string by commas and spaces
                        #print(box_cls, value)
                        pattern = r" 00\d+"
                        if re.search(r'.* (00\d+)', value):
                            final_value = re.split(pattern, value)[-1]
                            if len(final_value) > 0:
                                final_value=final_value.split()[0]
                        elif re.search(r'.* (\d+)$', value):
                            final_value = re.search(r'.* (\d+)$', value).groups()[-1]
                        #final_value = exval[-1] if exval[-1] else ''
                    elif box_cls == 16:
                        vv=re.split(' TS', value)[-1].replace(" ","")                        
                        final_value = "TS "+ re.search(r'(\d*)',vv).groups()[-1]#value.split(' TS ')[-1]
                        #print(value, final_value) 

                elif box_cls in (801,802,803,804,805,806,807,808,809,810,811,812):
                    if value.count(" ") >4:
                        final_value=value.split(" ", 5)[-1].strip().replace(" ","::")

                elif box_cls == 9:
                   # print("Valuessssss.... 9 :",value)
                    for k,val in value.items():
                        final_info_dict[k]={"Value":val, "Coordinates":coords}                        
                        #print(value)  
                    
                elif box_cls==10:
                    if re.search(r'.* (\d{4}[-]\d{2}[-]\d{2})', value):
                        exval = re.search(r'.* (\d{4}[-]\d{2}[-]\d{2})', value).groups()
                        final_value = exval[-1] if exval[-1] else ''
                    # else:
                    #     final_value = ''                        
                else:
                    reference_sentence = "Partner's name and address Nom et adresse de |'associe /associe \"/'associe Last name (print) Nom de famille (en lettres moulees) - First name Prenom Initials Initiales Filer's and address et adresse du declarant Tax shelter identification number (see statement on back Numero dinscription I'abri fiscal (lisez |enonce au dos"
                    # Split the sentences into words
                    reference_words = set(reference_sentence.split())
                    second_words = value.split()

                    # Remove matching words from the second sentence
                    filtered_words = [word for word in second_words if word not in reference_words]

                    # Join the filtered words back into a sentence
                    final_value = ' '.join(filtered_words)
                    final_value = final_value.replace(";","").replace(":","").replace("_","").replace("~","")
                
                if key != 9:
                    final_info_dict[key] = {"Value":final_value, "Coordinates":coords}
            else:
                  final_info_dict[key] = {"Value":'', "Coordinates":[]}
                
        output_json = self.order_box8(final_info_dict)
        # reference_words = set(output_json["Partner's name"]["Value"].split())
        # second_words = output_json["Partner's address"]["Value"].split()
        # filtered_words = [word for word in second_words if word not in reference_words]
        # output_json["Partner's address"] = {"Value":' '.join(filtered_words),"Coordinates":output_json["Partner's address"]["Coordinates"]}
        del output_json['Left Box Case Values']
        return output_json
    
    def process_single_pdf_t5013(self):
        if self.convert_pdf_to_image():
            class_info_dict = self.extract_identified_text(predictor)
            final_json_dict = self.mapping_data(class_info_dict)
            return final_json_dict