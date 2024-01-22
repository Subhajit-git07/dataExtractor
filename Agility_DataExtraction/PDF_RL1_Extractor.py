import fitz
import os
import cv2
import re
import json
import pickle
import easyocr
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from detectron2.engine import DefaultPredictor


reader = easyocr.Reader(["en"], gpu=False)

class PDF_RL1_ImageExtractor:
    """
    page_number=1 (Page Number to convert into image)
    """

    def __init__(self, pdf_file_path, output_image_path, template_json):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = 0.9
        self.page_number = 1
        self.template_json = template_json
        self.cfg_save_path = "/pas-models/RL1-Model/model.pkl"
        self.model_final_pth = "/pas-models/RL1-Model/model_final.pth"

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
        box_array = np.array(list(boxes_list), dtype=object)

        class_wise_coord_dict = {}
        for box_cord, score_val, pred_cls_val in zip(box_array, scores_list, pred_classes_list):
            bbox = box_cord.detach().cpu().numpy()
            if class_wise_coord_dict.get(pred_cls_val):
                if float(score_val) > class_wise_coord_dict[pred_cls_val]["score"]:
                    class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score": float(score_val)}
            else:
                class_wise_coord_dict[pred_cls_val] = {"bbox": bbox, "score": float(score_val)}

        pred_cls_data_dict = {}
        for pred_cls, bbox_scores_dict in class_wise_coord_dict.items():
            x_min, y_min, x_max, y_max = list(map(int, bbox_scores_dict["bbox"]))
            cropped_image = im[y_min:y_max, x_min:x_max, :]
            coords = [x_min, y_min, x_max, y_max]
            result = reader.readtext(cropped_image, detail=0, paragraph=False)
            result_2 = reader.readtext(cropped_image)
            para_res = ' '.join(result)

            pred_cls_data_dict[pred_cls] = ([para_res, result, result_2], coords)

        class_info_dict = {}

        for cls_pred, data_text_tup in pred_cls_data_dict.items():
            class_info_dict[cls_pred] = data_text_tup
        print("text_extraction is completed....!")
        return class_info_dict

    @staticmethod
    def get_address(value_2):
        try:
            grouped_data = {}
            for coords, text, _ in value_2:
                y_coord = coords[1][1]  # Using the second point's Y-coordinate for grouping
                found = False
                for grouped_y, texts in grouped_data.items():
                    if abs(grouped_y - y_coord) < 10:
                        texts.append(text)
                        found = True
                        break
                if not found:
                    grouped_data[y_coord] = [text]

            merged_grouped_data = {}
            for y_coord, texts in grouped_data.items():
                merged_text = ' '.join(texts)
                merged_grouped_data[y_coord] = merged_text

            keywords_to_remove = ['Prenom', 'Appartement', 'Numero', 'postale', 'village', 'Province',
                                  'Code postal']
            final_address = ""
            for y_coord, text in merged_grouped_data.items():
                if not any(keyword.lower() in text.lower() for keyword in keywords_to_remove):
                    final_address += f"{text}\n"


        except Exception as e:
            print("Exception :", e)
            final_address = ""
        return final_address

    def mapping_data(self, class_info_dict):
        final_info_dict = {}
        for box_cls, key in self.template_json.items():
            try:
                box_cls = int(box_cls)
            except:
                pass
            if 4 <= box_cls <= 26 and class_info_dict.get(box_cls):
                value = ""
                value_list, coords = class_info_dict.get(box_cls)
                value = value_list[0]
                if value[0] == "0":
                    value = value[1:]
                if ", " in value or " ," in value or " , " in value:
                    value = value.replace(", ", ",").replace(" ,", ",").replace(" , ", ",")
                if "1-" in value or "1 -" in value:
                    value = value.replace("1-", "I-").replace("1 -", "I-")
                numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', value)
                if len(numbers) >= 1:
                    final_value = numbers[-1]
                else:
                    # print("type", type(value))
                    # print("value doesnt match the pattern")
                    # print("value", value)
                    final_value = ""

                final_info_dict[key] = {"Value": final_value, "Coordinates": coords}
            elif box_cls == 0:
                value = ""
                value_list, coords = class_info_dict.get(box_cls)
                value = value_list[0]
                pattern = r'\b\d{4}\b'  # Looks for exactly four digits surrounded by word boundaries
                match = re.search(pattern, value)
                if match:
                    value = match.group()
                else:
                    value = ""
                final_value = value
                final_info_dict[key] = {"Value": value, "Coordinates": coords}
            elif box_cls in (1, 2, 3):
                try:
                    value_list, coords = class_info_dict.get(box_cls)
                    value = value_list[0]
                except:
                    pass
                #                 print("value_split", box_cls, value.split())
                #                 value = ""
                #                 value, coords = class_info_dict.get(box_cls)
                if len(value) >= 1:
                    value_list = value.split()
                    #                     print("value_list", value_list, "value_last", [value_list[-1]])
                    if len(value_list) >= 1:
                        if box_cls == 1 and "rele" in value_list[-1]:
                            value = ""
                        elif box_cls == 2 and "transm" in value_list[-1]:
                            value = ""
                        elif box_cls == 3 and (
                                "case" in value_list[-1] or "0)" in value_list[-1] or "o)" in value_list[-1].lower()):
                            value = ""
                        else:
                            value = value.split()[-1]

                final_info_dict[key] = {"Value": value, "Coordinates": coords}

            elif box_cls in (27, 28, 29, 30):
                try:
                    value_list, coords = class_info_dict.get(box_cls)
                    value = value_list[0]
                #                     print("value", value)
                except Exception as e:
                    print("Exception", e)
                    value = ""
                    coords = [0, 0, 0, 0]
                if len(value) >= 1:
                    if "0-" in value or "0 -" in value:
                        value = value.replace("0-", "O-").replace("0 -", "O-")
                    if " s" in value[-2:].lower() or " es" in value[-3:]:
                        value = value[:-2]
                    final_info_dict[key] = {"Value": value, "Coordinates": coords}
                else:
                    final_info_dict[key] = {"Value": "", "Coordinates": coords}

            elif box_cls in (33, 35, 36):
                try:
                    value_list, coords = class_info_dict.get(box_cls)
                    value = value_list[1]
                    # print("value_33", value)
                except Exception as e:
                    print("Exception", e)
                    value = ""
                    coords = [0, 0, 0, 0]               
                if (len(value)>1 and box_cls in (35, 36)) or (len(value)>2 and box_cls == 33):
                    final_output = value[-1]
                    if "famille" in final_output.lower():
                        final_output = ""
                    if "payeur" in final_output.lower():
                        final_output = ""
                    if "facult" in final_output.lower():
                        final_output = ""
                                                       
                    final_info_dict[key] = {"Value": final_output, "Coordinates": coords}
                else:
                    final_info_dict[key] = {"Value": "", "Coordinates": coords}

            elif box_cls == 34:
                try:
                    value_list, coords = class_info_dict.get(box_cls)
                    value = value_list[1]
                #                     print("value", value)
                except Exception as e:
                    print("Exception", e)
                    value = ""
                    coords = [0, 0, 0, 0]
                if len(value) >= 1:
                    words_to_check = ["assurance", "Numero", "sociale", "particul"]
                    for sublist in value[:]:
                        for word in words_to_check:
                            if word in sublist:
                                value.remove(sublist)
                                break
                    try:
                        final_info_dict[key] = {"Value": ' '.join(value), "Coordinates": coords}
                    except:
                        final_info_dict[key] = {"Value": value, "Coordinates": coords}


            else:
                try:
                    value_list, coords = class_info_dict.get(box_cls)
                    value = value_list[1]
                    value_2 = value_list[2]
                    final_address = self.get_address(value_2)
                    if final_address == "":
                        # print("value_in_else", type(value), [value])
                        keywords_to_remove = ['Prenom', 'Appartement', 'Numero', 'postale', 'village', 'Province', 'Code postal']

                        filtered_data = [item for item in value if not any(keyword in item for keyword in keywords_to_remove)]
                        # print("filtered_data", filtered_data)

                        final_address = ' '.join(filtered_data)
                    try:
                        final_info_dict[key] = {"Value": final_address, "Coordinates": coords}
                    except:
                        final_info_dict[key] = {"Value": value, "Coordinates": coords}

                except Exception as e:
                    print("Exception", e)
                    words_to_check = ["assurance", "Numero", "sociale", "particul"]
                    value = ""
                    coords = [0, 0, 0, 0]
                    try:
                        final_info_dict[key] = {"Value": value, "Coordinates": coords}
                    except:
                        final_info_dict[key] = {"Value": value, "Coordinates": coords}
        #                 print("box_cls", box_cls, value)

        print("Mapping has been completed...!!")
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

    def process_single_pdf_RL1(self):
        class_info_dict = self.process_pdf_to_text()
        final_json_dict = self.mapping_data(class_info_dict)
        return final_json_dict