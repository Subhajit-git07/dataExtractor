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

class PDF_RL2_ImageExtractor:
    """
    Detectron2
    """
    def __init__(self, pdf_file_path, output_image_path, template_json):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = 0.9
        self.page_number = 1
        self.template_json = template_json
        self.cfg_save_path = "/pas-models/RL2-Model/IS_cfg.pickle"
        # self.cfg_save_path = "./pas-models/RL2-Model/IS_cfg_10_12_2023.pickle"
        self.model_final_pth = "/pas-models/RL2-Model/model_final.pth"
        # self.model_final_pth = "./pas-models/RL2-Model/model_final_10_12_2023.pth"
        self.benefic_dict = {
                               "Bénéficiaire Nom de famille":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Prénom":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Appartement":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Numéro":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Rue, case postale":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Ville, village ou municipalite":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Province":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                               "Bénéficiaire Code postal":{"Value":"", "Coordinates":[0, 0, 0, 0]}
                             }
        self.emett_dict = {
                            "Nom et adresse du payeur ou de l’Emetteur":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            # "Adresse Du Payeur ou De l’Emetteur":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            "l’Emetteur Appartement":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            "l’Emetteur Numéro":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            "l’Emetteur Rue, Case Postale":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            "l’Emetteur Ville, Village ou Municipalite":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            "l’Emetteur Province":{"Value":"", "Coordinates":[0, 0, 0, 0]},
                            "l’Emetteur Code Postal":{"Value":"", "Coordinates":[0, 0, 0, 0]}
        }

        

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
    def extract_block_data(pred_cls, uncleaned_data_tup, block_2425_list):
        uncleaned_data, coords = uncleaned_data_tup
        eflag = False
        if re.findall(r"([A-Za-z]{1}-[0-9]{1}[A-Za-z]{1} [0-9.,/-]+|[A-Za-z]{1}-[0-9]{1} [0-9.,/-]+|[0-9]{3} [0-9.,/-]+)", uncleaned_data, flags=re.I):
            uncleaned_data = re.findall(r"([A-Za-z]{1}-[0-9]{1}[A-Za-z]{1} [0-9.,/-]+|[A-Za-z]{1}-[0-9]{1} [0-9.,/-]+|[0-9]{3} [0-9.,/-]+)", uncleaned_data, flags=re.I)
            eflag = True
        # elif re.findall(r"\d{1,3} [0-9.,/]+", uncleaned_data, flags=re.I):
            # uncleaned_data = re.search(r"\d{1,3} [0-9.,/]+", uncleaned_data, flags=re.I)
            # eflag = True
        uncleaned_data_list = copy.deepcopy(uncleaned_data)
        while uncleaned_data_list and eflag:
            # print("UNCL", uncleaned_data_list)
            box, val = "", ""
            box, val_info = uncleaned_data_list[0].split()
            val = re.sub(r"[^0-9A-Za-z,.\s-]", "", val_info)
            box = re.sub(r"[^0-9A-Za-z-]+", "", box)
            if box and val:
                block_2425_list.append((box, val, coords))
            uncleaned_data_list = uncleaned_data_list[1:]

    @staticmethod
    def clean_value_info(pred_cls, uncleaned_data_tup):
        uncleaned_data, coords = uncleaned_data_tup
        data_text = uncleaned_data.strip()
        val = ""
        if pred_cls in (20,):
            val_info = re.search(r"revenus (.*)", data_text)
            if val_info:
                val = val_info.groups()[0]
            return [val, coords]
        elif pred_cls in (14,):
            val_info = re.search(r"RVER\)? ([0-9.,-]+)", data_text)
            if val_info:
                val = val_info.groups()[0]
            return [val, coords]
        else:
            data_text_list = data_text.split()[-1]
            val = re.sub(r"[^0-9.,/-]+", "", data_text_list)
            if not val:
                val = ""
                coords = [0, 0, 0, 0]
        return (val, coords)
    
    @staticmethod
    def filter_identity_info(pred_cls, uncleaned_data_tup):
        uncleaned_data, coords = uncleaned_data_tup
        val = ""
        if pred_cls in (15,):
            for txt in ["du payeur", "Nom", "name"]:
                match_info = re.search(r""+txt, uncleaned_data, flags=re.I)
                if match_info:
                    val = uncleaned_data[match_info.end():].replace("du payeur", "").replace(")", "")
                    val = re.sub(r""+txt, "", val, flags=re.I)
                    break
            else:
                val = uncleaned_data[:]

        elif pred_cls in (10,):
            for txt in ["Prenom", "name", "First"]:
                match_info = re.search(r""+txt, uncleaned_data, flags=re.I)
                if match_info:
                    val = uncleaned_data[match_info.end():].replace(txt, "").replace(")", "")
                    val = re.sub(r""+txt, "", val, flags=re.I)
                    break
                else:
                    val = uncleaned_data[:]
        elif pred_cls in (12,):
            for txt in ["moulees", "lettres", "famille"]:
                match_info = re.search(r""+txt, uncleaned_data, flags=re.I)
                if match_info:
                    val = uncleaned_data[match_info.end():].replace(txt, "").replace(")", "")
                    val = re.sub(r""+txt, "", val, flags=re.I)
                    break
            else:
                val = uncleaned_data[:]
        elif pred_cls in (11,):
            match_info = re.search(r"(\s[A-Za-z]+)$", uncleaned_data.strip(), flags=re.I)
            if match_info:
                val = match_info.groups()[0]
            else:
                val = uncleaned_data[:]
            val = re.sub(r"Initiales", "", val, flags=re.I)
        return (val, coords) 

    @staticmethod
    def segregate_same_line_data(process_data, strip_text):

        pop_index = None
        for idx, data_tup in enumerate(process_data):
            if data_tup[-2].find(strip_text)!=-1:   
                pop_index = copy.deepcopy(idx)
                break

        left_data = []
        right_data = []
        if pop_index is not None:
            pop_tup = process_data[pop_index]
            x1, y2 = pop_tup[0][0][0], pop_tup[0][0][-1]
            for row_tup in process_data:
                test_x1, test_y1 = row_tup[0][0][0], row_tup[0][-2][-1]
                # print([x1, y2, test_x1, test_y1])
                if test_x1 < x1 and test_y1 > y2:
                    left_data.append(row_tup[-2])
                if test_x1>x1 and test_y1 > y2:
                    right_data.append(row_tup[-2])
        left_data = ' '.join(left_data)
        right_data = ' '.join(right_data)
        return left_data, right_data
    
    def get_cls8_name(self, pred_cls_data_details_dict, data_text_tup, cls_pred):
        name_params = ["Nom de famille", "Prenom", "Appartement",
                        "postale", "municipalite",
                        "Province"]
        
        info_tup = pred_cls_data_details_dict[cls_pred]
        cls_8_data_list, cls_8_coord_list = info_tup[0][1:], info_tup[1]

        for idx in range(1, len(name_params)):
            prev_text, curr_text = name_params[idx-1], name_params[idx]
            start_index, end_index = None, None
            for row_idx, row_tup in enumerate(cls_8_data_list, 1):
                tup_text = row_tup[-2]
                if tup_text.find(prev_text)!=-1:
                    start_index = copy.deepcopy(row_idx)
                elif tup_text.find(curr_text)!=-1:
                    end_index = copy.deepcopy(row_idx)
                    break
            # print("IDX", [idx, start_index, end_index])
            if start_index and end_index:
                if idx==1:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    self.benefic_dict["Bénéficiaire Nom de famille"] = {"Value":val, "Coordinates":data_text_tup[1]}
                elif idx==2:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    self.benefic_dict["Bénéficiaire Prénom"] = {"Value":val, "Coordinates":data_text_tup[1]}
                elif idx==3:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    appt, nume = self.segregate_same_line_data(val_data, "Numero")    
                    self.benefic_dict["Bénéficiaire Appartement"] = {"Value":appt, "Coordinates":data_text_tup[1]}
                    self.benefic_dict["Bénéficiaire Numéro"] = {"Value":nume, "Coordinates":data_text_tup[1]}
                elif idx==4:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    val = re.sub(r"Ville|village|ou", "", val.strip()).replace(' , ', "").strip()
                    self.benefic_dict["Bénéficiaire Rue, case postale"] = {"Value":val, "Coordinates":data_text_tup[1]}
                elif idx==5:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    # val = re.sub(r"Ville|village|ou", "", val.strip())
                    # print("VVVVVV", val)
                    self.benefic_dict["Bénéficiaire Ville, village ou municipalite"] = {"Value":val, "Coordinates":data_text_tup[1]}
            if idx==5 and end_index:
                # print("MMMMM", cls_pred)
                province_info = cls_8_data_list[end_index:]
                province, code_postal = self.segregate_same_line_data(province_info, "Code")
                code_postal = re.sub(r'postal', '', code_postal).strip()
                self.benefic_dict["Bénéficiaire Province"] = {"Value":province, "Coordinates":data_text_tup[1]}
                self.benefic_dict["Bénéficiaire Code postal"] = {"Value":code_postal, "Coordinates":data_text_tup[1]}
        return
    
    def get_cls9_name(self, pred_cls_data_details_dict, data_text_tup, cls_pred):
        name_params = ["adresse du payeur", "Appartement",
                        "postale", "municipalite",
                        "Province"]
        
        info_tup = pred_cls_data_details_dict[cls_pred]
        cls_8_data_list, cls_8_coord_list = info_tup[0], info_tup[1]

        for idx in range(1, len(name_params)):
            prev_text, curr_text = name_params[idx-1], name_params[idx]
            start_index, end_index = None, None
            for row_idx, row_tup in enumerate(cls_8_data_list, 1):
                tup_text = row_tup[-2]
                if tup_text.find(prev_text)!=-1:
                    start_index = copy.deepcopy(row_idx)
                elif tup_text.find(curr_text)!=-1:
                    end_index = copy.deepcopy(row_idx)
                    break
            # print("IDX", [idx, start_index, end_index])
            if start_index and end_index:
                if idx==1:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    self.emett_dict["Nom et adresse du payeur ou de l’Emetteur"] = {"Value":val, "Coordinates":data_text_tup[1]}
                elif idx==2:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    appt, nume = self.segregate_same_line_data(val_data, "Numero")    
                    self.emett_dict["l’Emetteur Appartement"] = {"Value":appt, "Coordinates":data_text_tup[1]}
                    self.emett_dict["l’Emetteur Numéro"] = {"Value":nume, "Coordinates":data_text_tup[1]}
                elif idx==3:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    val = re.sub(r"Ville|village|ou", "", val.strip()).replace(' , ', "").strip()
                    self.emett_dict["l’Emetteur Rue, Case Postale"] = {"Value":val, "Coordinates":data_text_tup[1]}
                elif idx==4:
                    val_data = cls_8_data_list[start_index:end_index-1]
                    val = ' '.join([ele[-2] for ele in val_data])
                    self.emett_dict["l’Emetteur Ville, Village ou Municipalite"] = {"Value":val, "Coordinates":data_text_tup[1]}
            if idx==4 and end_index:
                province_info = cls_8_data_list[end_index:]
                province, code_postal = self.segregate_same_line_data(province_info, "Code")
                code_postal = re.sub(r'postal', '', code_postal).strip()
                self.emett_dict["l’Emetteur Province"] = {"Value":province, "Coordinates":data_text_tup[1]}
                self.emett_dict["l’Emetteur Code Postal"] = {"Value":code_postal, "Coordinates":data_text_tup[1]}
        return

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
        pred_cls_data_details_dict = {}
        for pred_cls, bbox_scores_dict in class_wise_coord_dict.items():
            x_min, y_min, x_max, y_max = list(map(int, bbox_scores_dict["bbox"]))
            # if pred_cls in (16,):
                # ydiff = y_max - y_min
                # y_min = int(y_min + (ydiff * 0.32))
            cropped_image = im[y_min:y_max, x_min:x_max, :]
            coords = [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]
            result = reader.readtext(cropped_image, detail=0, paragraph=False)
            para_res = ' '.join(result)
            # print(pred_cls, '-->',  para_res)
            result1 = reader.readtext(cropped_image)
            # print("RESULT1", result1)
            pred_cls_data_dict[pred_cls] = (para_res, coords)
            pred_cls_data_details_dict[pred_cls] = [result1, coords]
            # print("##########################################################################\n")

        # SystemExit()
        sorted_class = sorted(pred_cls_data_dict)
        class_info_dict = {}
        block_2425_list = []
        for cls_pred in sorted_class:  
            data_text_tup = pred_cls_data_dict[cls_pred]
            try:
                if cls_pred in (9,):
                    # print("PRED", [cls_pred])
                    self.extract_block_data(pred_cls, data_text_tup, block_2425_list)
                elif cls_pred in (0, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20):
                    # print("DD", [data_text_tup[0]])
                    final_val = self.clean_value_info(cls_pred, data_text_tup)
                    # print("FFFFFFFF", final_val)
                    class_info_dict[cls_pred] = final_val
                elif cls_pred in (21,):
                    rel_data = re.search(r"\s[A-Za-z]{1}$", data_text_tup[0].strip())
                    if rel_data:
                        rel = rel_data.group()
                    class_info_dict[cls_pred] = (rel, data_text_tup[1])
                elif cls_pred in (22,):
                    year = re.sub(r"[^0-9]+", "", data_text_tup[0])
                    class_info_dict[cls_pred] = (year, data_text_tup[1])
                elif cls_pred in (8,):
                    self.get_cls8_name(pred_cls_data_details_dict, data_text_tup, cls_pred)
                    # print(self.benefic_dict)
                elif cls_pred in (1,):
                    self.get_cls9_name(pred_cls_data_details_dict, data_text_tup, cls_pred)
                    # print(self.emett_dict)
            except:pass
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
                elif box_cls in (9,):
                    if not block_2425_list:
                        block_2425_list = [('', '', [0, 0, 0, 0]) for _ in range(1, 4)]
                    if len(block_2425_list) < 4:
                        block_2425_list.extend([ ('', '', [0, 0, 0, 0]) for _ in range(4-len(block_2425_list))])
                    for idx, box_val in enumerate(block_2425_list, 1):
                        box, val, crd = box_val
                        if not val:
                            val = ""
                        row_key_info = " ".join([row_text, str(idx)])
                        # box_row_info = ' '.join([row_text, f"{box}"])
                        val = ' '.join([box, val]).strip()
                        final_info_dict[row_key_info] = {"Value":val, "Coordinates":crd}
                else:
                    final_info_dict[row_text] = {"Value": "", "Coordinates":(0, 0, 0, 0)}
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
    
    def process_single_pdf_rl2(self):
        try:
            class_info_dict, block_2425_list = self.process_pdf_to_text()
        except:
            class_info_dict, block_2425_list  = {}, []
        final_json_dict = self.mapping_data(class_info_dict, block_2425_list)
        final_json_dict.update(self.benefic_dict)
        final_json_dict.update(self.emett_dict)
        # print(self.emett_dict)
        try:
            del final_json_dict["name"]
            del final_json_dict["address"]
        except:
            pass
        return final_json_dict

if __name__ == '__main__':
    pdf_file_path = "Input/RL2/RL-2_Scanned_Sample 3.pdf"
    output_image_path = "output/images/rl2.png"
    template_json = {
            "22":"Année",
            "21":"Code du Relevé",
            "20":"Provenance des Revenus",
            "16":"No du dernier relevé transmis",
            "0":"Box A. Prestations D’un RPA",
            "3":"Box B. Prestations (REER, FERR, RPDB ou RPAC/RVER) ou Rentes",
            "4":"Box C. Autres  Paiements",
            "5":"Box D. Remboursement De Primes Au Conjoint Survivant (REER)",
            "6":"Box E. Prestation Réputée Reçue Au Décès (REER, FERR ou RPAC/RVER)",
            "7":"Box F. Remboursement De Cotisations Inutilisées (REER ou RPAC/RVER)",
            "10":"Box G. Montant Imposable En raison De La Révocation (REER, FERR ou RVDAA)",
            "11":"Box H. Autres Revenus (REER ou FERR)",
            "12":"Box I. Montant Donnant Droit à Une Déduction (REER ou FERR)",
            "13":"Box J. Impôt Du Québec Retenu à La Source",
            "14":"Box K. Revenus Gagnés Après Le Décès (REER, FERR ou RPAC/RVER)",
            "15":"Box L. Retrait Dans Le Cadre Du REEP",
            "17":"Box M. Montants Libérés D’impôt",
            "19":"Box O. Retrait Dans Le Cadre Du RAP",
            "9":"Renseignements Complémentaires Box",
            "2":"Numéro D’Assurance Sociale Du Bénéficiaire",
            "18":"Conjoint Cotisant (REER ou FERR) N- Numéro d‘assurance sociale",
            "8":"name",
            "1":"address"
              }

    rl2_obj = PDF_RL2_ImageExtractor(pdf_file_path, output_image_path, template_json)
    final_json = rl2_obj.process_single_pdf_rl2()
    # print(len(final_json))

    json_object = json.dumps(final_json, indent=4)
    
# Writing to sample.json
    with open("rl2_scanned.json", "w") as outfile:
        outfile.write(json_object)

    # for k, v in final_json.items():
    #     print("KEY", [k])
    #     print("VAL", [v])
    #     print("\n\n")