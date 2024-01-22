import os
# import concurrent.futures
import copy
# import sys
import json
import shutil


from PyPDF2 import PdfReader, PdfWriter
import fitz

from digital_form_identification import DigitalDocIdentificationPDF
from scanned_doc_identification import FormRecognizer_LayoutLM
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class DocProcess:
    def __init__(self, doc_path, output_dir):
        self.doc_path = doc_path
        self.output_dir = output_dir
        self.file_name = os.path.basename(self.doc_path)
        self.file_basname, self.file_ext = os.path.splitext(self.file_name)
        # self.split_page_folder = os.path.join(os.getcwd(),
        #                                      "split_doc_pages", 
        #                                      self.file_basname)
        # if not os.path.exists(self.split_page_folder):
        #     os.makedirs(self.split_page_folder)

        self.form_page_count = {

            "1099misc": 1,

            "1099int": 1,

            "t4a": 1,

            "t3": 1,

            "t4e": 1,

            "1098": 1,

            "1098c": 1,

            "1099div": 1,

            "1099b": 1,

            "1098e": 1,

            "1098t": 1,

            "1099oid": 1,

            "t4": 1,

            "t5": 1,

            "RL1": 3,

            "RL2": 3,

        }

        self.scan_skip_forms = {"K1", "k3"}

    @staticmethod
    def get_file_basename(file_path):
        file_name = os.path.basename(file_path)
        file_basename, file_ext = os.path.splitext(file_name)
        return file_basename, file_ext

    def form_page_path(self, file_page):
        page_path = os.path.join(self.split_page_folder, file_page)
        return page_path

    def split_pdf_pages(self):
        page_path_dict = {}
        read_obj = PdfReader(self.doc_path)
        self.no_of_pages = len(read_obj.pages)
        for page_num in range(self.no_of_pages):
            pdf_writer = PdfWriter()
            pdf_writer.add_page(read_obj.pages[page_num])

            doc_page_name = f"page_{page_num}.pdf"
            doc_page_path = os.path.join(self.split_page_folder, doc_page_name)

            with open(doc_page_path, 'wb') as out:
                pdf_writer.write(out)

            page_path = self.form_page_path(doc_page_name)
            page_path_dict[page_num] = page_path
        # read_obj.close()
        return page_path_dict, self.no_of_pages

    @staticmethod
    def identify_page(page_path):
        doc = fitz.open(page_path)
        page = doc.load_page(0)
        text = page.get_text()
        iflag = "digital"
        if len(text) < 100:
            iflag = "scanned"
        return iflag

    def identity_pdf_pages(self, all_pages_dict):
        iden_info_dict = {}
        digital_page_list = []
        scanned_page_list = []
        for page_num, page_path in all_pages_dict.items():
            page_identity = self.identify_page(page_path)
            iden_info_dict[page_num] = page_identity  # {"path": page_path, "identity": page_identity}
            if page_identity == "digital":
                digital_page_list.append(page_path)
            else:
                scanned_page_list.append(page_path)
        return digital_page_list, scanned_page_list, iden_info_dict

    @property
    def get_splited_pages_directory(self):
        all_pages_path = []
        for file_page in os.listdir(self.split_page_folder):
            page_path = os.path.join(self.split_page_folder, file_page)
            all_pages_path.append(page_path)
        return all_pages_path

    def segregate_form_page_number(self, page_wise_form_dict, dig_scan_page_map, all_digital_pages, all_scanned_pages):
        form_wise_page_info = {}
        unidentified_pages = {}
        form_type_info = ""
        for page_num in sorted(page_wise_form_dict):
            form_type_tup = page_wise_form_dict[page_num]
            form_type = form_type_tup[0]
            dig_scan_type = dig_scan_page_map[page_num]
            if form_type != "Unidentified" and not form_wise_page_info.get(form_type):
                form_type_info = copy.deepcopy(form_type)
                form_wise_page_info.setdefault(form_type_info, {}).setdefault(1, []).append(
                    (page_num, dig_scan_type, form_type_tup[1]))
            elif form_type == "Unidentified" and form_type_info:
                form_wise_page_info.setdefault(form_type_info, {}).setdefault(
                    max(form_wise_page_info.get(form_type_info, 1)), []).append(
                    (page_num, dig_scan_type, form_type_tup[1]))
                # unidentified_pages.append((page_num, dig_scan_type))
                unidentified_pages[page_num] = dig_scan_type
            elif form_type == form_type_info:
                num_page_list = form_wise_page_info.get(form_type_info, 1)
                if form_type in self.form_page_count:
                    page_cnt = self.form_page_count[form_type]
                    if len(num_page_list) >= page_cnt:
                        form_wise_page_info.setdefault(form_type_info, {}).setdefault(
                            max(form_wise_page_info.get(form_type_info, 0)) + 1, []).append(
                            (page_num, dig_scan_type, form_type_tup[1]))  ## initial
                    else:
                        form_wise_page_info.setdefault(form_type_info, {}).setdefault(
                            max(form_wise_page_info.get(form_type_info, 1)), []).append(
                            (page_num, dig_scan_type, form_type_tup[1]))
            elif form_type != form_type_info and form_type != "Unidentified":
                form_type_info = copy.deepcopy(form_type)
                form_wise_page_info.setdefault(form_type_info, {}).setdefault(
                    max(form_wise_page_info.get(form_type_info, 0)) + 1, []).append(
                    (page_num, dig_scan_type, form_type_tup[1]))
            elif form_type == "Unidentified":
                unidentified_pages[page_num] = dig_scan_type

        final_page_form_dict = {}
        for form_typ, no_of_forms_dict in form_wise_page_info.items():
            if form_typ in self.scan_skip_forms or form_typ not in self.form_page_count:  #
                # if form_typ not in self.scan_skip_forms or form_typ not in self.form_page_count:
                final_page_form_dict[form_typ] = no_of_forms_dict
            else:
                for form_no, page_list in no_of_forms_dict.items():
                    final_page_form_dict.setdefault(form_typ, {})[form_no] = page_list[:self.form_page_count[form_typ]]
                    # print("PPP", page_list[self.form_page_count[form_typ]:])
                    unidentified_pages.update(
                        {kv_tup[0]: kv_tup[1] for kv_tup in page_list[self.form_page_count[form_typ]:]})

        # for k, v in final_page_form_dict.items():
        # print(k, v)

        form_meta_dict = {

            "filename": self.file_name,
            "total_pages": self.no_of_pages,
            "page_type": {"digital": all_digital_pages, "scanned": all_scanned_pages},
            "Unidentified": {str(index + 1): {"form_name": "Unidentified",
                                              "start_page": key + 1,
                                              "end_page": key + 1,
                                              "confidence": 100,
                                              "doc_type": value}
                             for index, (key, value) in enumerate(unidentified_pages.items())}
            # "Unidentified":{k+1:v for k, v in unidentified_pages.items()}

        }

        print("final_page_form_dict", final_page_form_dict)
        for form_name, no_form_dict in final_page_form_dict.items():
            for form_no, page_list in no_form_dict.items():
                # print(form_name, form_no, page_list)
                doc_type = "digital"
                if any("scanned" in elem for elem in page_list):
                    doc_type = "scanned"
                if form_name in ("RL1", "w2", "W2", "RL-1"):
                    doc_type = "scanned"
                start_page, end_page = page_list[0][0], page_list[-1][0]
                start_conf = page_list[0][-1]
                form_meta_dict.setdefault(form_name, {})[form_no] = {"form_name": form_name,
                                                                     "start_page": start_page + 1,
                                                                     "end_page": end_page + 1,
                                                                     "confidence": start_conf,
                                                                     "doc_type": doc_type,
                                                                     }

        for k, v in form_meta_dict.items():
            print(k, '-->', v)
        return form_meta_dict

    def write_to_json(self, json_dict):
        json_object = json.dumps(json_dict, indent=4)
        with open(os.path.join(self.output_dir, self.file_basname + ".json"), "w") as outfile:
            outfile.write(json_object)

    def whole_pdf_page_indentify(self):
        pdf_page_iden_dict = {}
        digital_page_list = []
        scanned_page_list = []
        self.no_of_pages = 0
        with fitz.open(self.doc_path) as document:
            for page_number, page in enumerate(document, 0):
                words = page.get_text()
                if len(words) < 100:
                    form_type = "scanned"
                    pdf_page_iden_dict[page_number] = form_type
                    scanned_page_list.append(page_number)
                else:
                    form_type = "digital"
                    digital_page_list.append(page_number)
                    pdf_page_iden_dict[page_number] = form_type
                self.no_of_pages += 1
        return digital_page_list, scanned_page_list, pdf_page_iden_dict

    def create_images_from_pdf(self, page_number_list=[], page_path_dict={}):
        page_image_dict = {}
        if page_number_list:
            base_filename, ext = self.get_file_basename(self.doc_path)
            image_dir = os.path.join(self.output_dir, "page_images", base_filename)
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            dpi = 300
            pdf_document = fitz.open(self.doc_path)
            no_of_pages = pdf_document.page_count
            for page in range(no_of_pages):
                if page not in page_number_list:
                    continue
                first_page = pdf_document.load_page(page)
                pix = first_page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                page_image_path = os.path.join(image_dir, f"{page}.png")
                pix.save(page_image_path)
                page_image_dict[page] = page_image_path
            pdf_document.close()
        return page_image_dict, image_dir

    def filter_scan_parse_pages(self, page_classify_dict):
        scanned_page_list = []
        scan_skip_flag = False
        sorted_pages = sorted(page_classify_dict)
        for page_num in sorted_pages:
            form_name = page_classify_dict[page_num]
            if form_name[0] in self.scan_skip_forms:
                scan_skip_flag = True
            elif form_name[0] != "Unidentified" and form_name[0] not in self.scan_skip_forms:
                scan_skip_flag = False
            elif form_name[0] == "Unidentified" and not scan_skip_flag:
                scanned_page_list.append(page_num)
        return scanned_page_list

    @staticmethod
    def update_json_structure(original_json):
        key_mappings = {
            '1098': "1098",
            '1098c': "1098 C",
            '1098e': "1098-E",
            '1098t': "1098-T",
            '1099b': "1099-B",
            '1099div': "1099-DIV",
            '1099int': "1099-INT",
            '1099misc': "1099-MISC",
            '1099oid': "1099-OID",
            'benifitcharge': "benifitcharge",
            'generic_doc': "generic_doc",
            'Invoices': "Invoices",
            'K1': "K1",
            'k3': "K3",
            'RL1': 'RL-1',
            'RL2': 'RL-2',
            't3': "T3",
            't4': "T4",
            't4a': 'T4A',
            't4e': "T4E",
            'T4ps': "T4PS",
            't4rsp': "T4RSP",
            't5': "T5",
            'w2': "W2"

        }
        transformed_json = {
            'filename': original_json['filename'],
            'total_pages': original_json['total_pages'],
            'page_type': {
                'digital': original_json['page_type']['digital'],
                'scanned': original_json['page_type']['scanned']
            }
        }

        for key, value in original_json.items():
            if key not in ['filename', 'total_pages', 'page_type']:
                transformed_key = key if key not in key_mappings else key_mappings[key]

                for page, info in value.items():
                    if transformed_key not in transformed_json:
                        transformed_json[transformed_key] = {}
                    if page not in transformed_json[transformed_key]:
                        transformed_json[transformed_key][page] = {
                            'form_name': transformed_key,
                            'start_page': int(page),
                            'end_page': int(page),
                            'confidence': info['confidence'],
                            'doc_type': info['doc_type']
                        }
                    else:
                        transformed_json[transformed_key][page]['end_page'] = int(page)
                        transformed_json[transformed_key][page]['confidence'] = max(
                            transformed_json[transformed_key][page]['confidence'],
                            info['confidence']
                        )

        return transformed_json

    def executing_file(self):
        logging.info(f'Process Started after imports {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        digital_page_list, scanned_page_list, digital_scanned_page_wise = self.whole_pdf_page_indentify()
        logging.info(f'after identification digital or scanned. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        all_digital_pages = [elem + 1 for elem in digital_page_list]
        all_scanned_pages = [elem + 1 for elem in scanned_page_list]

        ddi_pdf = DigitalDocIdentificationPDF(self.doc_path)
        logging.info(f'Digital object creation. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        page_classify_dict = ddi_pdf.process_pdf(digital_page_list)
        logging.info(f'Digital classification. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        scanned_page_parse_list = self.filter_scan_parse_pages(page_classify_dict)
        scanned_page_list = scanned_page_list + scanned_page_parse_list
        logging.info(
            f'Identified the scanned list to be sent to layout LM. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # scanned_pred_list = []
        if scanned_page_list:
            page_path_dict, all_page_images = self.create_images_from_pdf(scanned_page_list)
            logging.info(f'Pdf to image split only scanned pages. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            # print("converted to images", page_path_dict)
            ################## Call scanned ##################################
            form_recognizer = FormRecognizer_LayoutLM(
                confidence_threshold=0.90,
            )
            logging.info(f'layout LM class object creation. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            for page_no, page_image_path in page_path_dict.items():
                form_recognizer.page_image_path = page_image_path
                result = form_recognizer.recognize_form()
                logging.info(f'For each scanned page time taken. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                result["page_no"] = page_no + 1
                # print(result)
                # print(result["page_types"][0])
                # scanned_info_dict = result["page_types"][0]
                # if scanned_info_dict["conf_score"] > 0.96:
                result["page_no"] = result["page_no"] - 1
                result["conf_score"] = int(result["conf_score"] * 100)
                page_classify_dict[page_no] = (result["page_class"], result["conf_score"])

            if os.path.exists(all_page_images):
                shutil.rmtree(all_page_images)
        # print({k+1: v for k, v in page_classify_dict.items()})
        # print(page_classify_dict)
        # print("Digital Prediction", page_classify_dict, '\n\n')
        # print("Scanned Prediction", scanned_pred_list, '\n\n')

        # print("\n\n", "Before Seg", page_classify_dict, "\n\n")
        logging.info(f'End JSON preparation start. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        seg_json = self.segregate_form_page_number(page_classify_dict,
                                                   digital_scanned_page_wise,
                                                   all_digital_pages,
                                                   all_scanned_pages)  ## Segregate code

        final_json = self.update_json_structure(seg_json)
        logging.info(f'End JSON preparation complete. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        return final_json




