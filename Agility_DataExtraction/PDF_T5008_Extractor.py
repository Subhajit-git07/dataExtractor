import fitz
import numpy as np
import cv2 as cv
import os
import pytesseract
import cv2
import pandas as pd
import time
import pickle
import re
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from detectron2.engine import DefaultPredictor
from collections import OrderedDict
import easyocr

cfg_save_path = "/pas-models/T5008-Model/IS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = "/pas-models/T5008-Model/model_final.pth" #os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

reader = easyocr.Reader(["en"], gpu=False)

class PDF_T5008_ImageExtractor:
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
            coords = [x_min, y_min, x_max, y_max]
            cropped_image = im[y_min:y_max+10, x_min:x_max+10, :]
            # cropped_image = cv2.resize(cropped_image, (x_max+10, y_max+10))
            # cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)
            grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            _, cropped_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            result = reader.readtext(cropped_image)
            para_res = ''
            for detection in result:
                para_res = para_res+detection[1]+' '
            
            pred_cls_data_dict[pred_cls] = (para_res.strip(), coords)

        class_info_dict = {}
        class_info_dict = pred_cls_data_dict
        
        return class_info_dict

    def mapping_data(self, class_info_dict):
        final_info_dict = {}
        for box_cls, key in self.template_json.items():
            box_cls = int(box_cls)
            if class_info_dict.get(box_cls):
                value, coords =  class_info_dict.get(box_cls)
                final_value = ""

                if box_cls in (7,14,16,17):
                    reference_sentence = "Prenam securtles Last name (print) Nom de famille (en lettres moulees} First name and initials Prenom et initiales privacy notice on your return: VOID ANNULE Identification of securities Designation des titres 24 Identification of securities received on settlement Designation des titres recus en de reglement (0n lettres moulees) Prenom et initiales Name and address of trader or dealer in securities Nom et adresse du negociant ou du courtier en valeurs Identification of securities Designation des titres 24 Identification of securities received on settlement Designation des titres recus en de reglement guise"
                    # Split the sentences into words
                    reference_words = set(reference_sentence.split())
                    second_words = value.split()

                    # Remove matching words from the second sentence
                    filtered_words = [word for word in second_words if word not in reference_words]

                    # Join the filtered words back into a sentence
                    final_value = ' '.join(filtered_words)
                    final_value = final_value.replace("17 ", "")

                elif box_cls in (4,18):
                    if re.search(r'.* (\d{2}[-]\d{2})', value):
                        exval = re.search(r'.* (\d{2}[-]\d{2})', value).groups()
                        final_value = exval[-1] if exval[-1] else ''
                    elif re.search(r'.* (\d{4})', value):
                        exval = re.search(r'.* (\d{4})', value).groups()
                        final_value = exval[-1] if exval[-1] else ''
                    else:
                        final_value = ''

                elif box_cls in (9,10,11):
                    if re.search(r'.* ([-][0-9,.]+|[0-9,.]+)',value):
                        exval = re.search(r'.* ([-][0-9,.]+|[0-9,.]+)',value).groups()
                        final_value = exval[-1] if exval[-1] else ''

                elif box_cls in (1,6,13):
                    if re.search(r'.* ([0-9,]+)', value):
                        exval = re.search(r'.* ([0-9,]+)', value).groups()
                        final_value = exval[-1] if exval[-1] else ''

                elif box_cls in (0,3):
                    reference_sentence = "13 currency Devises etrangeres Foreign 10 Report code Code du feuillet"
                    reference_words = set(reference_sentence.split())
                    second_words = value.split()
                    filtered_words = [word for word in second_words if word not in reference_words]
                    final_value = ' '.join(filtered_words)

                elif box_cls in (2,5,8,12):
                    key_lst = ['12 Recipient identification number Numero didentification du beneficiaire', '15 Type code of securities Code de genre de titres', '18 ISINCUSIP number Numero ISINICUSIP', '22 Type code of securities received on settlement Code de genre de titres recus en de reglement','ISINICUSIP number Numero ISINICUSIP','Type code ot secuities received &n settlement Coda de ganre de bbes fecus en guise de reglement','15 Type code al securities Code de gervre de itres','18 ISINCUSIP number Numero ISINCUSIP','12 Recipient Identiiication number Numero didentification du berreficiaire','22 Type code of securities received on settlement Code de genre de titres recus en guise de reglement','22 Type code of securities received o settlement Code de genre de titres recus en guise de reglement']
                    for x in key_lst:
                        if re.search(r''+x, value):
                            remaining_text = value.split(x)[1]
                            final_value = remaining_text.replace(' ', '').replace('guise', '').replace(')', '').replace(']','')
                else:
                    final_value = ''

                final_info_dict[key] = {"Value":final_value, "Coordinates":coords}
            else:
                  final_info_dict[key] = {"Value":'', "Coordinates":[]}
        return final_info_dict
    
    def process_single_pdf_t5008(self):
        if self.convert_pdf_to_image():
            class_info_dict = self.extract_identified_text(predictor)
            final_json_dict = self.mapping_data(class_info_dict)
            return final_json_dict




