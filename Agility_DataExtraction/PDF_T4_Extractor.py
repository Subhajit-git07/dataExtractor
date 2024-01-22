import fitz
import re
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
import json

import easyocr

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode 

from detectron2.engine import DefaultPredictor
import pickle

import random
import cv2

reader = easyocr.Reader(["en"], gpu=False)

class PDF_T4_ImageExtractor:
    """
    Detectron2
    """
    def __init__(self, pdf_file_path, output_image_path, template_json):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = 0.9
        self.page_number = 1
        self.template_json = template_json
        self.cfg_save_path = "/pas-models/T4-Model/IS_cfg.pickle"
        self.model_final_pth = "/pas-models/T4-Model/model_final.pth"

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
    def extract_24_and_25_block(pred_cls, uncleaned_data_tup, block_2425_list):
        uncleaned_data, coords = uncleaned_data_tup
        eflag = False
        if re.findall(r"\d{1,3} [0-9.,/]+", uncleaned_data, flags=re.I):
            uncleaned_data = re.findall(r"\d{1,3} [0-9.,/]+", uncleaned_data, flags=re.I)
            eflag = True
        # elif re.findall(r"\d{1,3} [0-9.,/]+", uncleaned_data, flags=re.I):
            # uncleaned_data = re.search(r"\d{1,3} [0-9.,/]+", uncleaned_data, flags=re.I)
            # eflag = True
        uncleaned_data_list = copy.deepcopy(uncleaned_data)
        while uncleaned_data_list and eflag:
            # print("UNCL", uncleaned_data_list)
            box, val_info = uncleaned_data_list[0].split()
            val = re.sub(r"[^0-9A-Z,.\s]", ".", val_info)
            box = re.sub(r"[^0-9]+", "", box)
            block_2425_list.append((box, val, coords))
            uncleaned_data_list = uncleaned_data_list[1:]

    @staticmethod
    def clean_value_info(pred_cls, uncleaned_data_tup):
        uncleaned_data, coords = uncleaned_data_tup
        data_list = uncleaned_data.strip().split()
        val = ""
        if pred_cls in (14,):
            val = data_list[-1]
            return (val, coords)
        if len(data_list)>=2:
            val = ' '.join(data_list[1:])
            val = re.sub(r"[^0-9A-Za-z,.\s]", ".", val)
        if val:
            val = f"{val}"
            return (val, coords)
        return (val, [0, 0, 0, 0])
    
    @staticmethod
    def filter_identity_info(pred_cls, uncleaned_data_tup):
        uncleaned_data, coords = uncleaned_data_tup
        val = ""
        if pred_cls in (22,):
            match_info = re.search(r"EMPLOYEUR", uncleaned_data, flags=re.I)
            if match_info:
                val = uncleaned_data[match_info.end():].replace("EMPLOYEUR", "")
                val = re.sub(r"EMPLOYEUR", "", val, flags=re.I)
        elif pred_cls in (19,):
            # print("Uncleaned data", uncleaned_data)
            #match_info = re.search(r"TEMPLOYEUR", uncleaned_data)
            val_str = uncleaned_data.split()[-1]
            if val_str:
                val = re.search(r'[0-9]+$', val_str.strip())
                if val:
                    val = val.group()
        elif pred_cls in (23,):
            match_info = re.search(r"INITIALE", uncleaned_data, flags=re.I)
            if match_info:
                val = uncleaned_data[match_info.end():].replace("INITIALE", "")
                val = re.sub(r"INITIALE", "", val, flags=re.I)
        return (val, coords) 

    def extract_identified_text(self, predictor):
        im = cv2.imread(self.output_image_path)
        outputs = predictor(im)
        boxes_list = outputs["instances"].pred_boxes
        scores_list = outputs["instances"].scores.detach().cpu().numpy()
        pred_classes_list = outputs["instances"].pred_classes.detach().cpu().numpy()

        box_array = np.array(list(boxes_list))#.detach().cpu().numpy()
        # print(box_array)

        class_wise_coord_dict = {}
        for box_cord, score_val, pred_cls_val in zip(box_array, scores_list, pred_classes_list):
                # print(box_cord, score_val, pred_cls_val)
            bbox = box_cord.detach().cpu().numpy()
            if class_wise_coord_dict.get(pred_cls_val):
                if float(score_val) >  class_wise_coord_dict[pred_cls_val]["score"]:
                    class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score":float(score_val)}
            else:
                class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score":float(score_val)}
        
        # print(class_wise_coord_dict)
        pred_cls_data_dict = {}
        for pred_cls, bbox_scores_dict in class_wise_coord_dict.items():
            x_min, y_min, x_max, y_max = list(map(int, bbox_scores_dict["bbox"]))
            cropped_image = im[y_min:y_max, x_min:x_max, :]
            coords = [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]
            result = reader.readtext(cropped_image, detail=0, paragraph=False)
            para_res = ' '.join(result)
            # print(pred_cls, '-->',  para_res)
            pred_cls_data_dict[pred_cls] = (para_res, coords)

        sorted_class = sorted(pred_cls_data_dict)
        class_info_dict = {}
        block_2425_list = []
        for cls_pred in sorted_class:  
            data_text_tup = pred_cls_data_dict[cls_pred]
            try:
                if cls_pred in (24, 25):    
                    # print("PRED", [cls_pred])
                    self.extract_24_and_25_block(pred_cls, data_text_tup, block_2425_list)
                elif cls_pred in (15, 0, 13, 2, 9, 3, 5, 4, 6, 10, 11, 7, 14, 8, 16, 18, 17, 20, 21, 1):
                    # print("DDDD", data_text_tup)
                    final_val = self.clean_value_info(cls_pred, data_text_tup)
                    # if cls_pred in (13, 6):
                        # print("VVVV", [final_val])
                    if not final_val:
                        continue
                    class_info_dict[cls_pred] = final_val
                elif cls_pred in (26,):
                    year = re.sub(r"[^0-9]+", "", data_text_tup[0])
                    class_info_dict[cls_pred] = (year, data_text_tup[1])
                elif cls_pred in (19, 22, 23):
                    final_val = self.filter_identity_info(cls_pred, data_text_tup)
                    class_info_dict[cls_pred] = final_val
            except:
                class_info_dict[cls_pred] = ('', [0, 0, 0, 0])
        return class_info_dict, block_2425_list 

    def mapping_data(self, class_info_dict, block_2425_list):
        final_info_dict = {}
        # print("BLOCK", block_2425_list)
        for box_cls, row_text in self.template_json.items():
            try:
                box_cls = int(box_cls)
            except:pass
            try:
                if class_info_dict.get(box_cls):
                    value, crds = class_info_dict.get(box_cls)
                    if not value:
                        value = ""
                    final_info_dict[row_text] = {"Value":value.strip(), "Coordinates":crds}
                elif box_cls in (24, 25):
                    # if not block_2425_list:
                        # block_2425_list = [('', '', [0, 0, 0, 0]) for _ in range(1, 7)]
                    if not block_2425_list:
                        block_2425_list = [('', '', [0, 0, 0, 0]) for _ in range(1, 7)]
                    if len(block_2425_list) < 6:
                        block_2425_list.extend([('', '', [0, 0, 0, 0]) for _ in range(6-len(block_2425_list))])
                    for idx, box_val in enumerate(block_2425_list, 1):
                        box, val, crd = box_val
                        if not val:
                            val = ""
                        box_row_info = ''.join([row_text, "Box ", str(idx)])
                        amt_row_info = ''.join([row_text, "Amount ", str(idx)])
                        final_info_dict[box_row_info] = {"Value":box, "Coordinates":crd}
                        final_info_dict[amt_row_info] = {"Value":val.strip(), "Coordinates":crd}
            except:pass
        return final_info_dict

    def loading_model_config(self):
        with open(self.cfg_save_path, 'rb') as f:
            cfg = pickle.load(f)
        return cfg

    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            cfg = self.loading_model_config()

            cfg.MODEL.WEIGHTS = self.model_final_pth#os.path.join(self.model_final_pth, 'model_final.pth')
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            predictor = DefaultPredictor(cfg)

            class_info_dict, block_1415_list = self.extract_identified_text(predictor)
            return class_info_dict, block_1415_list
        return {}, []
    
    def process_single_pdf_t4(self):
        try:
            class_info_dict, block_2425_list = self.process_pdf_to_text()
        except:
            class_info_dict, block_2425_list  = {}, []
        final_json_dict = self.mapping_data(class_info_dict, block_2425_list)
        return final_json_dict

if __name__ == '__main__':
    pdf_file_path = "Input/Sample 7 - T4 (Scanned) 1.pdf"
    output_image_path = "output/images/T4.png"
    template_json = {
             "22": "Employer's name",
             "26": "Year",
             "2": "Box 14. Employment Income",
             "9": "Box 22. Income Tax Deducted",
             "15": "Box 45. Employer-offered dental benefits",
             "19": "Box 54. Employer's Account Number",
             "3": "Box 16. Employee's CPP Contributions",
             "5": "Box 17. Employee's QPP Contributions",
             "0": "Box 10. Province of Employment",
             "1": "Box 12. Social Insurance Number",
             "13": "Box 29. Employment Code",
             "6": "Box 17A. Employee's second QPP contributions – see over",
             "4": "Box 16A. Employee's second CPP contributions– see over",
             "23": "Employee's address",
             "10": "Box 24. EI Insurable Earnings",
             "11": "Box 26. CPP/QPP Pensionable Earnings",
             "7": "Box 18. Employee's EI Premiums",
             "14": "Box 44. Unioin Dues",
             "8": "Box 20. RPP Contributions",
             "16": "Box 46. Charitable Donations",
             "18": "Box 52. Pension Adjustment",
             "17": "Box 50. RPP or DPSP Registration Number",
             "20": "Box 55. Employee's PPIP Premiums",
             "21": "Box 56. PPIP Insurable Earnings",
             "24": "Other Information - ",
             "25": "Other Information - "
              }

    t4_obj = PDF_T4_ImageExtractor(pdf_file_path, output_image_path, template_json)
    final_json = t4_obj.process_single_pdf_t4()
    print(final_json)

    json_object = json.dumps(final_json, indent=4)
 
# Writing to sample.json
    with open("t4_scanned.json", "w") as outfile:
        outfile.write(json_object)

    # for k, v in final_json.items():
    #     print("KEY", [k])
    #     print("VAL", [v])
    #     print("\n\n")