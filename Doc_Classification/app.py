import json
from flask import Flask, request, jsonify
import os
import time
import shutil
import logging
from datetime import datetime
from split_doc_pages import DocProcess
# import standard_json
import api_logic
from api_logic import BaseConfigDev as BaseConfig

# from api_logic import BaseConfigUAT as BaseConfig
# from api_logic import BaseConfigPROD as BaseConfig

app = Flask(__name__)


@app.route('/hi')
def index():
    return 'Hi, this is a extrcation code for scanned forms.'


@app.route('/', methods=['POST'])
def docClassification():
    if request.method == "POST":

        input_param = request.get_json(force=True)
        app.logger.info(f"Incoming Request Body:: {input_param}")
        if "inputdir" in input_param and "outputdir" in input_param:

            input_dir = input_param['inputdir']
            output_dir = input_param['outputdir']
            transaction_id = input_param['TransactionId']
            documents = input_param['Documents']

            app.logger.info(f"tran ::{transaction_id}")
            app.logger.info(f"doc ::{len(documents)}")

            if transaction_id or len(documents) == 0:
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                if os.path.exists(input_dir):
                    if len(os.listdir(input_dir)) >= 1:
                        final_output = []
                        consolidated_json = []
                        for document in documents:
                            document_id = document["DocumentId"]
                            document_name = document["DocumentName"]

                            if document_name.lower().endswith(".pdf"):
                                logging.basicConfig(filename=os.path.join(output_dir, str(document_name[:-4]) + '.log'),
                                                    level=logging.INFO,
                                                    format='%(asctime)s - %(levelname)s - %(message)s')
                                logging.info(f'start. {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                                pdf_file_path = os.path.join(input_dir, document_name)
                                # output_images_path = os.path.join(output_dir, "split_image")

                                # if not os.path.exists(output_images_path):
                                # os.mkdir(output_images_path)

                                # output_image_path = os.path.join(output_images_path, f"{document_name[:-4]}.png")
                                # output_json_path = os.path.join(output_dir, f"{document_name[:-4]}.json")
                                output_json = {}
                                ########################### Doc classification code call #############################################
                                doc_obj = DocProcess(pdf_file_path, output_dir)
                                output_json = doc_obj.executing_file()

                                ## Writing to JSON
                                doc_obj.write_to_json(output_json)

                                app.logger.info(f"output_json :: {output_json}")

                                # Consolidate all the jsons before standard format
                                consolidated_json.append({"DocumentId": document_id,
                                                          "ExtractedData": output_json})

                                # [{}, {}]
                                # Call the standard JSON functions.
                                # output_json = standard_json.sub_dict_append(output_json)
                                # output_json = standard_json.append_upper_dict_new(output_json, append_json)

                                # Dumping The JSON
                                # with open(output_json_path, 'w', encoding='utf-8') as json_file:
                                # json.dump(output_json, json_file, indent=4)

                                # Delete the output image file
                                # if os.path.exists(output_images_path):
                                # shutil.rmtree(output_images_path)

                                final_output.append({
                                    "DocumentId": document_id,
                                    "DocumentName": document_name,
                                    "ExtractedData": output_json
                                })

                        # Final Output for api_logic

                        api_response = api_logic.run_api_logic(BaseConfig, input_param, consolidated_json)

                        app.logger.info(f"end_api_response :: {api_response}")

                        logger = logging.getLogger()
                        handlers = logger.handlers

                        for handler in handlers:
                            handler.close()
                            logger.removeHandler(handler)
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