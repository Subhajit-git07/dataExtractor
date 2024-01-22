import fitz
import numpy as np
import cv2 as cv
import easyocr
import os
import pytesseract
import cv2
import pandas as pd
import time
import pickle
import re
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from detectron2.engine import DefaultPredictor

reader = easyocr.Reader(["en"], gpu=False)

class PDF_T4e_ImageExtractor:
    """
    page_number=1 (Page Number to convert into image)
    """

    def __init__(self, pdf_file_path, output_image_path, template_json):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = 0.9
        self.page_number = 1
        self.template_json = template_json
        self.cfg_save_path = "/pas-models/T4e-Model/IS_cfg_T4e.pickle"
        self.model_final_pth = "/pas-models/T4e-Model/model_final_T4e.pth"

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


    @staticmethod
    def clean_value_info(pred_class, text_tup):
        uncleaned_data, coords = text_tup
        allowed_chars = r"0-9A-Za-z,.\[\]%\s"
        if pred_class in (10, 13):
            value = uncleaned_data.replace("_", "").replace("-", "").replace("= ", "").replace('“', "").replace('°', "").replace('"', "").replace('|', "").replace('  ', " ")
            keys = "RECIPIENT'S NAME AND ADDRESS NOM ET ADRESSE DU BENEFICIAIRE PAYER'S NAME NOM DU PAYEUR EL BENEFICIALRE PAYER RECIPIENT"
            for key in keys.split():
                value = value.replace(key, '')
            cleaned_value = re.sub(r'\s+', ' ', value).strip() 
        else:
            value = re.sub(f"[^{allowed_chars}]", "", uncleaned_data)
            value = value.replace("_","").replace("-","").replace("= ", "").replace('“',"").replace('°',"").replace('"', "").replace('|',"").replace('  '," ")
            cleaned_value = re.sub(r'[^0-9.,%]', ' ', value)
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
            
        return (cleaned_value, coords)

    def extract_identified_text(self, predictor):
        im = cv2.imread(self.output_image_path)
        outputs = predictor(im)
        boxes_list = outputs["instances"].pred_boxes
        scores_list = outputs["instances"].scores.detach().cpu().numpy()
        pred_classes_list = outputs["instances"].pred_classes.detach().cpu().numpy()
        box_array = np.array(list(boxes_list))
    
        class_wise_coord_dict = {}
        for box_cord, score_val, pred_cls_val in zip(box_array, scores_list, pred_classes_list):
            bbox = box_cord.detach().cpu().numpy()
            if class_wise_coord_dict.get(pred_cls_val):
                if float(score_val) >  class_wise_coord_dict[pred_cls_val]["score"]:
                    class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score":float(score_val)}
            else:
                class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score":float(score_val)}
        
        pred_cls_data_dict = {}
        for pred_cls, bbox_scores_dict in class_wise_coord_dict.items():
            x_min, y_min, x_max, y_max = list(map(int, bbox_scores_dict["bbox"]))
            cropped_image = im[y_min:y_max, x_min:x_max, :]
            coords = [x_min, y_min, x_max, y_max]
            result = reader.readtext(cropped_image, detail=0, paragraph=False)
            para_res = ' '.join(result).upper()
            
            pred_cls_data_dict[pred_cls] = (para_res, coords)

        class_info_dict = {}
        
        for cls_pred, data_text_tup in pred_cls_data_dict.items():
            final_val = self.clean_value_info(cls_pred, data_text_tup)
            class_info_dict[cls_pred] = final_val
#         print(class_info_dict)
        return class_info_dict

    def mapping_data(self, class_info_dict):
        final_info_dict = {}
        for box_cls, key in self.template_json.items():
            box_cls = int(box_cls)
            if class_info_dict.get(box_cls):
                value, coords =  class_info_dict.get(box_cls)
                final_value = ""
                if box_cls in (0,1,2,3,4,5):
                    key1 = "Box/Case "+str(box_cls+1)
                    key2 = "Amount/Montant "+str(box_cls+1)
                    if re.search(r'(\d{2}) ([0-9,.]+)', value):
                        final_value = re.search(r'(\d{2}) ([0-9,.]+)', value).groups()
                        val1 = final_value[0] if final_value[0] else ''
                        val2 = final_value[1] if final_value[1] else ''
                    elif re.search(r'(\d{2})', value):
                        final_value = re.search(r'(\d{2})', value).groups()
                        val1 = final_value[0] if final_value[0] else ''
                        val2 = ''
                    elif re.search(r'([0-9,.]+)', value):
                        final_value = re.search(r'([0-9,.]+)', value).groups()
                        val1 = ''
                        val2 = final_value[0] if final_value[0] else ''
                    else:
                        val1 = ''
                        val2 = ''
                    final_info_dict[key1] = {"Value":val1, "Coordinates":coords}
                    final_info_dict[key2] = {"Value":val2, "Coordinates":coords}
                else:
                    pat14 = re.search(r'(\d+ [%]|\d+[%])', value)
                    pat7 = re.search(r'(\d{3} \d{3} \d{3}|\d{9})', value)
                    pato1 = re.search(r'.* ([0-9,.]+)$', value)
                    pato2 = re.search(r'^([0-9,.]+)$', value)
                    if box_cls==14 and pat14:
                        final_value = pat14.groups()[-1]
                    elif box_cls==7 and pat7:
                        final_value = pat7.groups()[-1]
                    elif box_cls in (10,13):
                        final_value = value
                    else:
                        if len(value.split())>1 and pato1:
                            final_value = pato1.groups()[-1]
                        else:
                            final_value = pato2.groups()[-1] if pato2 else ''
                            if re.search(r'^\d{2}$', final_value):
                                final_value = ''
                    final_info_dict[key] = {"Value":final_value, "Coordinates":coords}
            else:
                if box_cls in (0,1,2,3,4,5):
                    key1 = "Box/Case "+str(box_cls+1)
                    key2 = "Amount/Montant "+str(box_cls+1)
                    final_info_dict[key1] = {"Value":'', "Coordinates":[]}
                    final_info_dict[key2] = {"Value":'', "Coordinates":[]}
                else:
                    final_info_dict[key] = {"Value":'', "Coordinates":[]}
        return final_info_dict

    def loading_model_config(self):
        with open(self.cfg_save_path, 'rb') as f:
            cfg = pickle.load(f)
        return cfg

    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            cfg = self.loading_model_config()
            cfg.MODEL.WEIGHTS = self.model_final_pth
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            predictor = DefaultPredictor(cfg)

            class_info_dict = self.extract_identified_text(predictor)

            return class_info_dict
        return {}, []

    def process_single_pdf_t4e(self):
        class_info_dict = self.process_pdf_to_text()
        final_json_dict = self.mapping_data(class_info_dict)
        return final_json_dict
