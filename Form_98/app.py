import os
import shutil
import time
import re
from flask import *
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from PDF_1098C_ImageExtractor import PDF_1098C_ImageExtractor
import requests


app = Flask(__name__)


def search_keyword_and_extract_value(array, keyword):
    try:
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)

        for element in array:
            match = keyword_pattern.search(element)
            if match:
                # Find the value starting from the keyword position
                value = element[match.end():]

                # Use regex to split by "\n\n" and remove leading/trailing spaces
                values = [v.strip() for v in re.split(r'\n\n', value) if v.strip()]

                if values:
                    return "\n\n".join(values)

        # Fuzzy matching for close proximity and spelling mistakes
        best_match_ratio = -1
        best_match_value = None

        for element in array:
            ratio = fuzz.ratio(keyword.lower(), element.lower())

            if ratio > best_match_ratio:
                best_match_ratio = ratio
                best_match_value = element

        if best_match_ratio > 80:  # Adjust the threshold as needed
            # Find the value starting from the best match position
            value = best_match_value[len(keyword):]

            # Use regex to split by "\n\n" and remove leading/trailing spaces
            values = [v.strip() for v in re.split(r'\n\n', value) if v.strip()]

            if values:
                return "\n\n".join(values)

        # If no direct or fuzzy match found, try to find the best fuzzy match in the array
        best_fuzzy_match = process.extractOne(keyword, array)
        if best_fuzzy_match[1] > 80:  # Adjust the threshold as needed
            best_fuzzy_match_value = best_fuzzy_match[0]
            # Find the value starting from the best fuzzy match position
            value = best_fuzzy_match_value[len(keyword):]

            # Use regex to split by "\n\n" and remove leading/trailing spaces
            values = [v.strip() for v in re.split(r'\n\n', value) if v.strip()]

            if values:
                return "\n\n".join(values)
    except Exception as e:
        pass

    # If there's an exception or no match found, return an empty string
    return ""

@app.route('/hi')
def index():
    return 'Hi, this is a extrcation code for 1098C scanned forms.'


@app.route('/', methods=['POST'])
def DT_1098C_Extraction():
    try:
        if request.method == "POST":
            start_time = time.time()
            input_param = request.get_json(force=True)
            app.logger.info(f"Incoming Request Body:: {input_param}")
            if "inputdir" in input_param and "outputdir" in input_param:
                input_dir = input_param['inputdir']
                output_dir = input_param['outputdir']
                transaction_id = input_param['TransactionId']
                documents = input_param['Documents']

                # Your template JSON
                with open('config.json', 'r', encoding='utf-8') as config_file:
                    config = json.load(config_file)
                template_json = config['template_json']
                threshold_contour = config['threshold_contour']
                threshold_centroid = config['threshold_centroid']
                page_number = config['page_number']


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
                            for document in documents:
                                document_id = document["DocumentId"]
                                document_name = document["DocumentName"]

                                if document_name.lower().endswith(".pdf"):
                                    pdf_file_path = os.path.join(input_dir, document_name)
                                    output_images_path = os.path.join(output_dir, "split_image")
                                    output_json = template_json.copy()
                                    if os.path.exists(output_images_path):
                                        pass
                                    else:
                                        os.mkdir(output_images_path)

                                    output_image_path = os.path.join(output_images_path, f"{document_name[:-4]}.png")
                                    output_json_path = os.path.join(output_dir, f"{document_name[:-4]}.json")

                                    pdf_extractor = PDF_1098C_ImageExtractor(pdf_file_path, output_image_path, threshold_contour, page_number, threshold_centroid)

                                    # Process the PDF to extract text data
                                    extracted_data = pdf_extractor.process_pdf_to_text()

                                    # Remove "$\n\n" from all elements in the array
                                    cleaned_array = [s.replace("$\n\n", "") for s in extracted_data]

                                    # Exclude elements with a length greater than 500 to exclude whole image extractions
                                    cleaned_array_processed = [element for element in cleaned_array if len(element) <= 500]

                                    app.logger.info(cleaned_array_processed)
                                    # print("cleaned_array_processed", cleaned_array_processed)
                                    # Create a new JSON object to store the extracted values
                                    final_json = {}

                                    # # Load the configuration file
                                    # with open('config.json', 'r', encoding='utf-8') as config_file:
                                    #     config = json.load(config_file)

                                    # Get the template JSON from the loaded configuration
                                    template_json = config['template_json']

                                    # Iterate through the keys of the template_json
                                    for key in template_json.keys():
                                        # Call the fuzzy search function for the current key
                                        result = search_keyword_and_extract_value(cleaned_array_processed, key)
                                        # print("key", key, "result", result)

                                        # If a result is found, update the final_json
                                        if result is not None:
                                            result = result.replace("$\n\n", "").replace("|", "").replace(",", "").replace(
                                                ".", "").replace("$", "").replace("\n", " ")
                                            final_json[key] = result
                                            # print("key", key, "result", result)

                                    with open(output_json_path, 'w') as json_file:
                                        json.dump(final_json, json_file, indent=4)

                                    # Delete the output image file
                                    if os.path.exists(output_images_path):
                                        shutil.rmtree(output_images_path)

                                    final_output.append({
                                        "DocumentId": document_id,
                                        "DocumentName": document_name,
                                        "ExtractedData": final_json
                                    })


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
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)