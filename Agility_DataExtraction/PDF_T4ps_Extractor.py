import fitz
import os
import cv2
import re
import json
import pickle
import easyocr
import pytesseract
import numpy as np
# from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from detectron2.engine import DefaultPredictor

reader = easyocr.Reader(["en"], gpu=False)

cfg_save_path = "/pas-models/T4ps-Model/t4ps_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = "/pas-models/T4ps-Model/model_final.pth" # os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

class PDF_T4ps_ImageExtractor:
    """
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path, template_json, page_number=1):
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
            
            # Load the image
            image = cv2.imread(self.output_image_path)
            cv2.imwrite(self.output_image_path, image)
            
            pdf_document.close()
            print(f'Conversion of the first page to {self.output_image_path} completed.')
            return True
        except Exception as e:
            print(f"Error: Pdf to Image Conversion failed!! {str(e)}")
            return False

    def extract_identified_text(self,image_path, predictor):
        im = cv2.imread(image_path)
        outputs = predictor(im)

        boxes_list = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores_list = outputs["instances"].scores.detach().cpu().numpy()
        pred_classes_list = outputs["instances"].pred_classes.detach().cpu().numpy()
        cropped_images_dict = {}
        
        for i, (box, score, pred_cls) in enumerate(zip(boxes_list, scores_list, pred_classes_list), start=1):
            x_min, y_min, x_max, y_max = map(int, box)

            coord_dict = {}
            cropped_image = im[y_min:y_max, x_min:x_max, :]

            #Cordinates
            x = x_min
            y = y_min
            w = (x_max - x_min)
            h = (y_max - y_min)
            
            # Use EasyOCR to extract text
            results = reader.readtext(cropped_image)
            concatenated_text = ' '.join([item[1] for item in results])
        
            coord_dict['Value'] = concatenated_text.replace("/", ".")
            coord_dict['Coordinates'] = (y, y + h, x, x + w)
            cropped_images_dict[pred_cls] = coord_dict
        
        return cropped_images_dict
    
    def mapping_data(self, class_info_dict):
        final_info_dict = {}
        for box_cls, key in self.template_json.items():
            box_cls = int(box_cls)
            if class_info_dict.get(box_cls):
                value =  class_info_dict.get(box_cls)['Value']
                coords = class_info_dict.get(box_cls)['Coordinates']
                final_value = ''
                
                if box_cls in (0,9,10,16,20):
                    value = value.replace("'","").replace(".","")
                    if box_cls==0 and re.search(r"(Employers .* Iemployeur|Employers .* employeur) (.*)",value):
                        grps = re.search(r"(Employers .* Iemployeur|Employers .* employeur) (.*)",value).groups()
                        final_value = grps[-1] if grps[-1] else value
                    elif box_cls==10 and value.split('Initiales ')[-1]:
                        final_value = value.split('Initiales ')[-1]
                    elif box_cls==16 and value.split('benefices ')[-1]:
                        final_value = value.split('benefices ')[-1]
                    elif box_cls==20:
                        year = re.search(r'.*(\d{4})', value)
                        final_value = year.groups()[-1] if year else ''
                    else:
                        final_value=value
                
                elif box_cls == 5:
                    final_value = re.sub(r'\d', '', value)
                    
                elif box_cls in (2,15):
                    if re.search(r'^(\d{2}) (.*)', value):
                        grps = re.search(r'^(\d{2}) (.*)', value).groups()
                        final_value = grps[-1] if grps[-1] else value
                        
                else:
                    if re.search(r'(\d{2}) ([0-9,.]+)', value):
                        grps = re.search(r'(\d{2}) ([0-9,.]+)', value).groups()
                        final_value = grps[-1] if grps[-1] else value
                
                final_info_dict[key] = {"Value":final_value, "Coordinates":coords}
                
            else:
                final_info_dict[key] = {"Value":"", "Coordinates":""}
        
        return final_info_dict
        
    def process_single_pdf_T4ps(self):
        if self.convert_pdf_to_image():
            cropped_images_dict = self.extract_identified_text(self.output_image_path, predictor)
            final_mapping = self.mapping_data(cropped_images_dict)
            return final_mapping

