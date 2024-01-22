import fitz
# Run OCR on the cropped images

import numpy as np
import cv2 as cv

import os
import pytesseract
import cv2
import time

import matplotlib.pyplot as plt

import re
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class PDF_T4rsp_ImageExtractor:
    
    def __init__(self, pdf_file_path, output_image_path, template_json, threshold=0.9, page_number=1,
                 centriod_closeness_threshold=90):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.page_number = page_number
        self.centriod_closeness_threshold = centriod_closeness_threshold
        self.template_json = template_json

        #The key_label_mapping is teh mapping file, based on the fuzzy search will be performed.
        self.key_label_mapping = {
                "Year": "Year",
                "Recipient's Last Name": "Recipient's name and address — Nom et adresse du bénéficiaire\nLast name First name Initials\nNom de famille Prénom Initiales",
                "Recipient's First Name": "Recipient's name and address — Nom et adresse du bénéficiaire\nLast name First name Initials\nNom de famille Prénom Initiales",
                "Recipient's Initials": "Recipient's name and address — Nom et adresse du bénéficiaire\nLast name First name Initials\nNom de famille Prénom Initiales",
                "Recipient's Address": "Recipient's name and address — Nom et adresse du bénéficiaire\nLast name First name Initials\nNom de famille Prénom Initiales",
                "Box 16. Annuity Payments": "Annuity Payments",
                "Box 18. Refund of Premiums": "Refund of Premiums",
                "Box 20. Refund of unused contributions": "Refund of unused\ncontributions",
                "Box 22. Withdrawal and Commutation Payments": "Withdrawal and\ncommutation payments",
                "Box 25. LLP Withdrawal": "LLP Withdrawal",
                "Box 26. Amounts Deemed Received on Deregistration": "Amounts deemed received\non deregistration",
                "Box 28. Other Income or Deductions": "Other Income or Deductions",
                "Box 30. Income Tax Deducted":"Income Tax Deducted",
                "Box 34. Amounts Deemed Received on Death": "Amounts deemed received\non death",
                "Box 37. Advanced Life Deferred Annuity Purchase":"Advanced Life Deferred Annuity Purchase",
                "Box 27. HBP Withdrawal":"HBP Withdrawal",
                "Box 35. Transfers on Breakdown":"Transfers on breakdown of\nmarriage or common-law part",
                "Box 24. Contributor Spouse or Common-Law Partner":"Contributor Spouse or Common-Law Partner",
                "Box 36. Contributor Spouse or Common-Law Partner Social Insurance Number":"Spouse's or common-law partner's\nsocial insurance number’\n",
                "Box 12. Social Insurance Number":"Social insurance number”\n",
                "Box 14. Contract Number":"Contract Number",
                "Box 60. Name of Payer of Plan":"Name of payer (issuer) of plan - Nom du payeur (émetteur) du régime\n",
                "Box 61. Account Number":"Account Number",
                "Box 40. Tax-Paid Amount":"Tax-Paid Amount"
        }

    def convert_pdf_to_image(self):
        try:
            pdf_document = fitz.open(self.pdf_file_path)
            dpi = 300
            first_page = pdf_document.load_page(0)
            pix = first_page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            pix.save(self.output_image_path)

            # Load the image
            image = cv2.imread(self.output_image_path)

            # Specify the upper and lower line coordinates (y-coordinates)
            upper_line_y = 60  # Adjust this value as needed
            lower_line_y = 1050  # Adjust this value as needed


            # Crop the image using the specified lines
            cropped_image = image[upper_line_y:lower_line_y, :]
            cv2.imwrite(self.output_image_path, cropped_image)

            pdf_document.close()
            print(f'Conversion of the first page to {self.output_image_path} completed.')
            return True
        except Exception as e:
            print(f"Error: Pdf to Image Conversion failed!! {str(e)}")
            return False

    def add_outer_bounding_box(self):
        try:
            image = cv2.imread(self.output_image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outermost_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(outermost_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.imwrite(self.output_image_path, image)
            print("Outer Bounding Box has been added and Image Generated")
        except Exception as e:
            print(f"Error: Outer Bounding Box generation Failed due to OpenCV error!! {str(e)}")

    def angle_cos(self,p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

    def find_squares(self):
        img = cv2.imread(self.output_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squares = []

        # Blur the image and Apply thresholding to enhance contrast
        blur = cv2.GaussianBlur(gray, (1, 1), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            if rect[2] < 15 or rect[3] < 15:
                continue

            x, y, w, h = rect
            area = w * h
            if area > 10000 and area < 1000000:
                squares.append(cnt) 

        print(f"Squares Detection Succesfull with Count: {len(squares)}")
        return squares
    
    def parse_address_string(self,address_string):
        # Split the address string into lines
        lines = address_string.split('\n')
        
        # Remove single character and empty elements
        lines = [item for item in lines if len(item) > 1]

        # Initialize variables to store the extracted information
        first_name = ""
        last_name = ""
        initials = ""
        recipient_address = ""
        
        
        # Process each line to extract information
        if len(lines) >= 3:
            full_name = lines[0].split()  # Split the full name into individual parts
            if len(full_name) >= 2:
                first_name = full_name[1]
                last_name = full_name[0]
                initials = full_name[2] if len(full_name) > 2 else ""
            
            
            if (len(lines) >= 3):
                recipient_address = lines[1] + lines[2]
            if (len(lines) >= 4):
                recipient_address = lines[1] + lines[2] + lines[3]
            if (len(lines) >= 5):
                recipient_address = lines[1] + lines[2] + lines[3] + lines[4]
            if (len(lines) >= 6):
                recipient_address = lines[1] + lines[2] + lines[3] + lines[4] + lines[5]
            

        # Return the extracted information as a dictionary
        return {
            "First Name": first_name,
            "Last Name": last_name,
            "Initials": initials,
            "Recipient's Address": recipient_address
        }
    
    def final_contour_data_extraction(self,grouped_squares):
        start_time = time.time()
        image = cv2.imread(self.output_image_path)
        data = []
        coord_dict = {}


        for i, contour in enumerate(grouped_squares):

            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            coord_dict = {}
            coord_dict['coordinates'] = (y, y + h, x, x + w)

            # Crop the image based on the bounding box
            cropped_image = image[y:y + h, x:x + w]


            #Optional: Display the image with its centroid for verification
#             plt.figure(figsize=(10, 10))
#             plt.imshow(cropped_image)
#             plt.show()

            text = pytesseract.image_to_string(cropped_image, lang='eng',config = "--psm 6")

            #print(text)
            coord_dict['text'] = text
            data.append(coord_dict)


        print("Text extraction completed.")
        end_time = time.time()
        print(f"Elapsed Time: {end_time - start_time}")
        return data
    
    def make_fuzzy_processed_array(self,dictionary_list):
        processed_array_for_fuzzy = []

        for item in dictionary_list:
            text_value = item.get('text', '').strip()  # Get the 'text' value and remove leading/trailing whitespace
            if text_value:  # Check if the text value is not empty
                processed_array_for_fuzzy.append(text_value)

        return processed_array_for_fuzzy
    
    def search_keyword_and_extract_value(self,array, keyword):
        try:
            keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)

            for element in array:
                match = keyword_pattern.search(element)
                if match:
                    # Find the value starting from the keyword position
                    value = element[match.end():].strip()

                    # Use regex to split by "\n" and remove leading/trailing spaces
                    values = [v.strip() for v in re.split(r'\n', value) if v.strip()]

                    if values:
                        return "\n".join(values)

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
                value = best_match_value[len(keyword):].strip()

                # Use regex to split by "\n" and remove leading/trailing spaces
                values = [v.strip() for v in re.split(r'\n', value) if v.strip()]

                if values:
                    return "\n".join(values)

            # If no direct or fuzzy match found, try to find the best fuzzy match in the array
            best_fuzzy_match = process.extractOne(keyword, array)
            if best_fuzzy_match[1] > 80:  # Adjust the threshold as needed
                best_fuzzy_match_value = best_fuzzy_match[0]
                # Find the value starting from the best fuzzy match position
                value = best_fuzzy_match_value[len(keyword):].strip()

                # Use regex to split by "\n" and remove leading/trailing spaces
                values = [v.strip() for v in re.split(r'\n', value) if v.strip()]

                if values:
                    return "\n".join(values)

        except Exception as e:
            pass

        # If there's an exception or no match found, return an empty string
        return ""
    
    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            self.add_outer_bounding_box()
            squares = self.find_squares()
            data = self.final_contour_data_extraction(squares)
            return data
        else:
            return []
    
    def process_single_pdf_T4rsp(self):

        extracted_data = self.process_pdf_to_text()
        cleaned_array = extracted_data.copy()


        # Exclude elements with length greater than 500 to exclude whole image extractions
        cleaned_array_processed = [d for d in cleaned_array if len(d.get('text', '')) <= 500]

        # Call the function to extract text values
        processed_array_for_fuzzy = self.make_fuzzy_processed_array(cleaned_array_processed)

        coordinates = ()
        final_json = {}
        phrases_to_check = ["Box 61. Account Number", "Recipient's First Name", "Recipient's Last Name", "Recipient's Initials", "Recipient's Address","Box 60. Name of Payer of Plan"]
        
        for sentence_key in self.template_json.keys():
            
            if not any(phrase in sentence_key for phrase in phrases_to_check):
                output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                output = ''.join([char if char.isdigit() or char == '-' or char == '.' or char == ',' else '' for char in output])
            elif "Recipient's First Name" in sentence_key:
                output_all = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                parsed_info = self.parse_address_string(output_all)
                output = parsed_info['First Name']
            elif "Recipient's Last Name" in sentence_key:
                output_all = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                parsed_info = self.parse_address_string(output_all)
                output = parsed_info['Last Name']
            elif "Recipient's Initials" in sentence_key:
                output_all = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                parsed_info = self.parse_address_string(output_all)
                output = parsed_info['Initials']
            elif "Recipient's Address" in sentence_key:
                output_all = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                parsed_info = self.parse_address_string(output_all)
                output = parsed_info["Recipient's Address"]
            elif "Box 60. Name of Payer of Plan" in sentence_key:
                output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                # Check if the string contains "\n"
                if(len(output)>2):
                    if "\n" in output:
                        output = output.split("\n")[1]
                    else:
                        output = output
                else:
                    output = ''
            elif "Box 61. Account Number" in sentence_key:
                output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key])
                # Check if the string contains "\n"
                if(output.count("\n") == 1):
                    output = output.split("\n")[0]
                else:
                    output = ''
            else:
                output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, self.key_label_mapping[sentence_key]).split("\n")[0]
                
                
                
                
            for d in cleaned_array_processed:
                for key, value in d.items():
                    sentence_extract = ""
                    if key == "coordinates":
                        coordinates = ()
                        coordinates = value
                    if key == "text" and coordinates:
                        sentence_extract = value
                    
#                     # If a result is found, update the final_json
                    if output is not None:
                        if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                            interim_dict = {}
                            interim_dict["Value"] = output
                            interim_dict["Coordinates"] = coordinates
                            final_json[sentence_key] = interim_dict

        return final_json