import fitz
import re
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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

class PDF_T3_ImageExtractor:
    """
    Detectron2
    """
    def __init__(self, pdf_file_path, output_image_path, template_json):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = 0.9
        self.page_number = 1
        self.template_json = template_json
        self.cfg_save_path = "/pas-models/T3-Model/IS_cfg.pickle"
        self.model_final_pth = "/pas-models/T3-Model/model_final.pth"

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
    def extract_14_and_15_block(pred_cls, uncleaned_data_tup, block_1415_list):
        uncleaned_data, coords = uncleaned_data_tup
        eflag = False
        if re.findall(r"\d{3} [0-9.,/]+", uncleaned_data, flags=re.I):
            uncleaned_data = re.findall(r"\d{3} [0-9.,/]+", uncleaned_data, flags=re.I)
            eflag = True
        while uncleaned_data and eflag:
            box, val_info = uncleaned_data[0].split()
            val = re.sub(r"[^0-9A-Z,.\s]", ".", val_info)
            box = re.sub(r"[^0-9]+", "", box)
            block_1415_list.append((box, val, coords))
            uncleaned_data = uncleaned_data[1:]

    @staticmethod
    def clean_value_info(pred_cls, uncleaned_data_tup):
        uncleaned_data, coords = uncleaned_data_tup
        data_list = uncleaned_data.strip().split()
        val = ""
        if pred_cls in (1,):
            val = ' '.join(data_list[-2:])
            return (val, coords)
        elif pred_cls in (18,):
            val1 = re.search(r'\d{4}', uncleaned_data)
            if val1:
                val = val1.group()
            return (val, coords)
        if len(data_list)>=2:
            val = ' '.join(data_list[1:])
            val = re.sub(r"[^0-9A-Z,.\s-]", ".", val)
        elif len(data_list)==1 and pred_cls not in (2,):
            val = data_list[-1].strip()
            val = re.sub(r"[^0-9A-Z,.\s-]", ".", val)
        return (val, coords)
    
    @staticmethod
    def filter_identity_info(pred_cls, uncleaned_data_tup):
        uncleaned_data, coords = uncleaned_data_tup
        val = ""
        if pred_cls in (16,):
            match_info = re.search(r"BENEFICIAIRE", uncleaned_data)
            if match_info:
                val = uncleaned_data[match_info.end():].replace("BENEFICIAIRE", "")
            else:
                val = uncleaned_data[:]
        elif pred_cls in (17,):
            match_info = re.search(r"FIDUCIE", uncleaned_data)
            if match_info:
                val = uncleaned_data[match_info.end():].replace("FIDUCIE", "")
            else:
                val = uncleaned_data[:]
        elif pred_cls in (13,):
            match_info = re.search(r"NOTES", uncleaned_data)
            if match_info:
                val = uncleaned_data[match_info.end():].replace("NOTES", "")
            else:
                val = uncleaned_data[:]
        return (val, coords) 

    def extract_identified_text(self, predictor):
        im = cv2.imread(self.output_image_path)
        outputs = predictor(im)
        boxes_list = outputs["instances"].pred_boxes
        scores_list = outputs["instances"].scores.detach().cpu().numpy()
        pred_classes_list = outputs["instances"].pred_classes.detach().cpu().numpy()

        box_array = np.array(list(boxes_list))#.detach().cpu().numpy()

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
            coords = [x_min, y_min, x_max, y_max]
            result = reader.readtext(cropped_image, detail=0, paragraph=False)
            para_res = ' '.join(result).upper()
            # print(pred_cls, '-->',  para_res)
            pred_cls_data_dict[pred_cls] = (para_res, coords)

        class_info_dict = {}
        block_1415_list = []
        sorted_class = sorted(pred_cls_data_dict)
        for cls_pred in sorted_class:
            data_text_tup = pred_cls_data_dict[cls_pred]
            try:
                if cls_pred in (14, 15):
                    self.extract_14_and_15_block(pred_cls, data_text_tup, block_1415_list)
                elif cls_pred in (10, 11, 12, 4, 7, 5, 8, 9, 6, 0, 1, 2, 3, 18):
                    # print("DDD", data_text_tup)
                    final_val = self.clean_value_info(cls_pred, data_text_tup)
                    class_info_dict[cls_pred] = final_val
                elif cls_pred in (13, 16, 17):
                    final_val = self.filter_identity_info(cls_pred, data_text_tup)
                    # print([cls_pred, final_val])
                    class_info_dict[cls_pred] = final_val
                elif cls_pred in (19,):
                    year, month = "", ""
                    year_data = re.search(r"MONTH (\d+)", data_text_tup[0])
                    month_data = re.search(r"MOIS (\d+)", data_text_tup[0])
                    if year_data:
                        year = year_data.groups()[0]
                    if month_data:
                        month = month_data.groups()[0]
                    class_info_dict[cls_pred] = ([year, month], data_text_tup[1])
            except:
                pass
        return class_info_dict, block_1415_list 

    def mapping_data(self, class_info_dict, block_1415_list):
        # print("block", block_1415_list)
        final_info_dict = {"Recipient's name (last name first)": {"Value":'', "Coordinates":[0,0,0,0]}}
        for box_cls, row_text in self.template_json.items():
            try:
                box_cls = int(box_cls)
            except:
                pass
            # if class_info_dict.get(box_cls):
            try:
                if box_cls in (19,):
                    ym, coords = class_info_dict[box_cls]
                    year, month = ym
                    if year:
                        row_info = ''.join([row_text, "year"])
                        final_info_dict[row_info] = {"Value":year.strip(), "Coordinates":coords}
                    else:
                        row_info = ''.join([row_text, "year"])
                        final_info_dict[row_info] = {"Value":'', "Coordinates":coords}
                    if month:
                        row_info = ''.join([row_text, "Month"])
                        final_info_dict[row_info] = {"Value":month.strip(), "Coordinates":coords}
                    else:
                        row_info = ''.join([row_text, "Month"])
                        final_info_dict[row_info] = {"Value":'', "Coordinates":coords}
                elif box_cls in (14, 15):
                    if not block_1415_list:
                        block_1415_list = [('', '', [0, 0, 0, 0]) for _ in range(1, 7)]
                    if len(block_1415_list) < 6:
                        block_1415_list.extend([('', '', [0, 0, 0, 0]) for _ in range(6-len(block_1415_list))])

                    for idx, box_val in enumerate(block_1415_list, 1):
                        box, val, crd = box_val
                        box_row_info = ''.join([row_text, str(idx)])
                        val_row_info = ''.join(["Amount", str(idx)])
                        final_info_dict[box_row_info] = {"Value":box, "Coordinates":crd}
                        final_info_dict[val_row_info] = {"Value":val.strip(), "Coordinates":crd}
                else:
                    value, coords =  class_info_dict.get(box_cls, ('', [0, 0, 0, 0]))
                    final_info_dict[row_text] = {"Value":value.strip(), "Coordinates":coords}
            except:
                pass
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
    
    def process_single_pdf_t3(self):
        try:
            class_info_dict, block_1415_list = self.process_pdf_to_text()
        except:
            class_info_dict, block_1415_list = {}, []
        final_json_dict = self.mapping_data(class_info_dict, block_1415_list)
        return final_json_dict

if __name__ == '__main__':
    pdf_file_path = "Input/t3/T3 2022 Scanned-9.pdf"
    # pdf_file_path = "Input/T3.pdf"
    output_image_path = "output/images/T3.png"
    template_json = {
              "18": "Year",
              "10": "Box 49. Actual Amount of Eligible Dividends",
              "11": "Box 50. Taxable Amount of Eligible Dividends",
              "12": "Box 51. Devidend Tax Credit for Eligible Dividends",
              "4": "Box 21. Capital Gains", 
              "7": "Box 30. Capital Gains Eligible for Deduction",
              "5": "Box 23. Actual Amount of Dividends..",
              "8": "Box 32. Taxable Amount of Dividends..",
              "9": "Box 39. Dividends Tax Credit for Dividends..",
              "6": "Box 26. Other Income",
              "19": "Trust year end ",
              "14": "Other Information. Box/Case ",
              "13": "Footnotes",
              "16": "Recipient's address", 
              "17": "Trust's name and address",
              "0": "Box 12. Recipient Identification Number",
              "1": "Box 14. Account Number",
              "2": "Box 16. Report Code",
              "3": "Box 18 . Beneficiary Code"

              }
    
    t3_obj = PDF_T3_ImageExtractor(pdf_file_path, output_image_path, template_json)
    final_json = t3_obj.process_single_pdf_t3()

    # for k, v in final_json.items():
    #     print("KEY", [k])
    #     print("VAL", [v])
    #     print('\n\n')

    json_object = json.dumps(final_json, indent=4)
 
# Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)

    print(len(final_json))


