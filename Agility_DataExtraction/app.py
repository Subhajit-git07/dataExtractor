
import json
from flask import Flask, request, jsonify
import os
import time
import shutil
from PDF_1098_Extractor import PDF_1098_ImageExtractor
from PDF_1098E_ImageExtractor import PDF_1098E_ImageExtractor
from PDF_1098T_ImageExtractor import PDF_1098T_ImageExtractor
from PDF_1098C_ImageExtractor import PDF_1098C_ImageExtractor
from PDF_1099DIV_Extractor import PDF_1099_DIV_ImageExtractor
from PDF_1099B_Extractor import PDF_1099B_ImageExtractor
from PDF_1099_MISC_ImageExtractor import PDF_1099_MISC_ImageExtractor
from PDF_1099INT_Extractor import PDF_1099_INT_ImageExtractor
from PDF_1099OID_ImageExtractor import PDF_1099_OID_ImageExtractor
from PDF_T5_ImageExtractor import PDF_T5_ImageExtractor
from PDF_W2_ImageExtractor import PDF_W2_ImageExtractor
from PDF_T4a_Extractor import PDF_T4a_ImageExtractor
from PDF_T4_Extractor import PDF_T4_ImageExtractor
from PDF_T3_Extractor import PDF_T3_ImageExtractor
from PDF_T4ps_Extractor import PDF_T4ps_ImageExtractor
from PDF_T4e_Extractor import PDF_T4e_ImageExtractor
from PDF_T4rsp_Extractor import PDF_T4rsp_ImageExtractor
from PDF_RL1_Extractor import PDF_RL1_ImageExtractor
from PDF_RL2_Extractor import PDF_RL2_ImageExtractor
from PDF_T5008_Extractor import PDF_T5008_ImageExtractor
from PDF_T5013_Extractor import PDF_T5013_ImageExtractor
import standard_json
import api_logic
# from api_logic import BaseConfigDev as BaseConfig
from api_logic import BaseConfigUAT as BaseConfig
# from api_logic import BaseConfigPROD as BaseConfig



app = Flask(__name__)


@app.route('/hi')
def index():
    return 'Hi, this is a extrcation code for scanned forms.'


@app.route('/', methods=['POST'])
def DT_1098_Extraction():
    if request.method == "POST":
        start_time = time.time()
        input_param = request.get_json(force=True)
        app.logger.info(f"Incoming Request Body:: {input_param}")
        if "inputdir" in input_param and "outputdir" in input_param:
            input_dir = input_param['inputdir']
            output_dir = input_param['outputdir']
            transaction_id = input_param['TransactionId']
            documents = input_param['Documents']

            form_type = input_param['form_type']

            # Your template JSON
            with open('config.json', 'r', encoding='utf-8') as config_file:
                config = json.load(config_file)
            
            threshold_contour = config['threshold_contour']
            threshold_centroid = config['threshold_centroid']
            page_number = config['page_number']
            threshold_cosine = config['threshold_cosine']
            append_json = config["append_json"]

            # print("threshold_cosine", threshold_cosine)


            app.logger.info(f"tran ::{transaction_id}")
            app.logger.info(f"doc ::{len(documents)}")

            if transaction_id or len(documents) == 0:
                if os.path.exists(output_dir):
                    pass
                else:
                    os.mkdir(output_dir)

                if os.path.exists(input_dir):
                    if len(os.listdir(input_dir)) >= 1:
                        final_output = []
                        consolidated_json = []
                        for document in documents:
                            document_id = document["DocumentId"]
                            document_name = document["DocumentName"]

                            if document_name.lower().endswith(".pdf"):
                                pdf_file_path = os.path.join(input_dir, document_name)
                                output_images_path = os.path.join(output_dir, "split_image")
                                

                                if os.path.exists(output_images_path):
                                    pass
                                else:
                                    os.mkdir(output_images_path)

                                output_image_path = os.path.join(output_images_path, f"{document_name[:-4]}.png")
                                output_json_path = os.path.join(output_dir, f"{document_name[:-4]}.json")
                                output_json = {}
                                ########################### Code for 1098 Form #############################################
                                if form_type == "1098":
                                    template_json = config['template_json_1098']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1098_ImageExtractor(pdf_file_path, output_image_path,
                                                                            template_json, threshold_contour,
                                                                            page_number, threshold_centroid, threshold_cosine)
                                    output_json = pdf_extractor.process_single_pdf_1098()

                                ########################### Code for 1098C Form #############################################
                                elif form_type == "1098 C":
                                    template_json = config['template_json_1098c']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1098C_ImageExtractor(pdf_file_path, output_image_path,template_json)
                                    output_json = pdf_extractor.process_single_pdf_1098C(pdf_file_path)

                                ########################### Code for 1098E Form #############################################
                                elif form_type == "1098-E":
                                    template_json = config['template_json_1098e']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1098E_ImageExtractor(pdf_file_path, output_image_path,template_json)
                                    output_json = pdf_extractor.process_single_pdf_1098E(pdf_file_path)
                                    # print(output_json)

                                ########################### Code for 1098T Form #############################################
                                elif form_type == "1098-T":
                                    template_json = config['template_json_1098t']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1098T_ImageExtractor(pdf_file_path, output_image_path,template_json)
                                    output_json = pdf_extractor.process_single_pdf_1098T(pdf_file_path)

                                ########################### Code for 1099DIV Form #############################################
                                elif form_type == "1099-DIV":
                                    template_json = config['template_json_1099div']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1099_DIV_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_1099_div()

                                elif form_type == "1099-B":
                                    template_json = config['template_json_1099B']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1099B_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_1099B()

                                elif form_type == "1099-MISC":
                                    template_json = config['template_json_1099_MISC']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1099_MISC_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_1099_misc()

                                ########################### Code for 1099INT Form #############################################
                                elif form_type == "1099-INT":
                                    template_json = config['template_json_1099int']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1099_INT_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_1099_int()

                                ########################### Code for 1099OID Form #############################################
                                elif form_type == "1099-OID":
                                    template_json = config['template_json_1099oid']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_1099_OID_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_1099_oid()

                                ########################### Code for T5 Form #############################################
                                elif form_type == "T5":
                                    template_json = config['template_json_t5']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_T5_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_T5()

                                ########################### Code for W2 Form #############################################
                                elif form_type == "W2":
                                    template_json = config['template_json_w2']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_W2_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_W2()

                                ########################### Code for T4A Form #############################################
                                elif form_type == "T4a" or form_type == "T4A":
                                    template_json = config['template_json_t4a']
                                    output_json = template_json.copy()
                                    pdf_extractor = PDF_T4a_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = pdf_extractor.process_single_pdf_t4a()

                                ########################### Code for T4 Form #############################################
                                elif form_type == "T4":
                                    template_json = config['template_json_t4']
                                    output_json = template_json.copy()
                                    t4_obj = PDF_T4_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = t4_obj.process_single_pdf_t4()

                                ########################### Code for T3 Form #############################################

                                elif form_type == "T3":
                                    template_json = config['template_json_t3']
                                    output_json = template_json.copy()
                                    t3_obj = PDF_T3_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = t3_obj.process_single_pdf_t3()
                                    # print(final_json)

                                ########################### Code for T4PS Form #############################################

                                elif form_type == "T4PS":
                                    template_json = config['template_json_t4ps']
                                    output_json = template_json.copy()
                                    t4ps_obj = PDF_T4ps_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = t4ps_obj.process_single_pdf_T4ps()
                                    # print(final_json)

                                ############################ Code for T4E Form #############################################

                                elif form_type == "T4E":
                                    template_json = config['template_json_t4e']
                                    output_json = template_json.copy()
                                    t4e_obj = PDF_T4e_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = t4e_obj.process_single_pdf_t4e()

                                ############################ Code for T4rsp Form #############################################

                                elif form_type == "T4RSP":
                                    template_json = config['template_json_t4rsp']
                                    output_json = template_json.copy()
                                    t4rsp_obj = PDF_T4rsp_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = t4rsp_obj.process_single_pdf_T4rsp()

                                    ############################ Code for RL1 Form #############################################

                                elif form_type == "RL-1":
                                    template_json = config['template_json_RL1']
                                    output_json = template_json.copy()
                                    RL1_obj = PDF_RL1_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = RL1_obj.process_single_pdf_RL1()

                                ############################ Code for RL2 Form #############################################

                                elif form_type == "RL-2":
                                    template_json = config['template_json_RL2']
                                    output_json = template_json.copy()
                                    RL2_obj = PDF_RL2_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = RL2_obj.process_single_pdf_rl2()

                                ############################ Code for T5008 Form #############################################

                                elif form_type == "T5008":
                                    template_json = config['template_json_T5008']
                                    output_json = template_json.copy()
                                    T5008_obj = PDF_T5008_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = T5008_obj.process_single_pdf_t5008()

                                ############################ Code for T5013 Form #############################################

                                elif form_type == "T5013":
                                    template_json = config['template_json_T5013']
                                    output_json = template_json.copy()
                                    T5013_obj = PDF_T5013_ImageExtractor(pdf_file_path, output_image_path, template_json)
                                    output_json = T5013_obj.process_single_pdf_t5013()

                                app.logger.info(f"output_json :: {output_json}")

                                # Consolidate all the jsons before standard format
                                consolidated_json.append({"DocumentId": document_id, "ExtractedData": output_json})

                                # Call the standard JSON functions.
                                output_json = standard_json.sub_dict_append(output_json)
                                output_json = standard_json.append_upper_dict_new(output_json, append_json)

                                #Dumping The JSON
                                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                                    json.dump(output_json, json_file, indent=4)

                                # Delete the output image file
                                if os.path.exists(output_images_path):
                                    shutil.rmtree(output_images_path)

                                final_output.append({
                                    "DocumentId": document_id,
                                    "DocumentName": document_name,
                                    "ExtractedData": output_json
                                })

                        # Final Output for api_logic
                        
                        api_response = api_logic.run_api_logic(BaseConfig, input_param, consolidated_json)

                        app.logger.info(f"end_api_response :: {api_response}")

                        # return final_output
                        return jsonify({
                                        "TransactionId": transaction_id,
                                        "Results": final_output
                                         }), 200

                    else:
                        return jsonify({'message': "Input directory is empty."}), 400

                else:
                    return jsonify({'message': "Input directory is not present"}), 400
            else:
                return jsonify({'message': "Transaction ID is blank or No document details were provided"}), 400

        else:
            return jsonify({'message': "Please pass the Input and Output directories"}), 400

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8000, debug=True)