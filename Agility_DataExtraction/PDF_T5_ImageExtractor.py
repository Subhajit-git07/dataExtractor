import fitz
import numpy as np
import cv2 as cv

import os
import pytesseract
import cv2
import pandas as pd
import time

import re
import os
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class PDF_T5_ImageExtractor:
    """
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path,template_json,page_number=1):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.page_number = page_number
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
            
            # Specify the upper and lower line coordinates (y-coordinates)
            upper_line_y = 62  # Adjust this value as needed
            lower_line_y = 1200  # Adjust this value as needed
            

            # Crop the image using the specified lines
            cropped_image = image[upper_line_y:lower_line_y, :]
            cv2.imwrite(self.output_image_path, cropped_image)
            
            pdf_document.close()
            print(f'Conversion of the first page to {self.output_image_path} completed.')
            return True
        except Exception as e:
            print(f"Error: Pdf to Image Conversion failed!! {str(e)}")
            return False
            
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
            #print(area
            if area > 25000 and area < 500000:
                squares.append(cnt) 
                

        print(f"Squares Detection Succesfull with Count: {len(squares)}")
        return squares


    def final_contour_data_extraction(self, grouped_squares):
        start_time = time.time()
        image = cv2.imread(self.output_image_path)
        data = []
        

        for i, contour in enumerate(grouped_squares):
            
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            coord_dict = {}
            coord_dict['coordinates'] = (y, y + h, x, x + w)

            # Crop the image based on the bounding box
            cropped_image = image[y:y + h, x:x + w]
            
            #Optional: Display the image with its centroid for verification
#             plt.figure(figsize=(5, 5))
#             plt.imshow(cropped_image)
#             plt.show()
            
#             area = w * h
#             print(area)
            #gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(cropped_image, lang='eng')
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

    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            squares = self.find_squares()
            data = self.final_contour_data_extraction(squares)
            if len(data) > 0:
                return data
            else:
                return []
            
    def search_keyword_and_extract_value(self,array, keyword):
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
    
    def is_similarity_above_threshold(self, string1, string2, threshold=80):
        similarity_score = fuzz.ratio(string1, string2)
        return similarity_score > threshold
    
    # Function to extract data from a single PDF file
    def process_single_pdf_T5(self):
        
        extracted_data = self.process_pdf_to_text()
        cleaned_array = extracted_data.copy()
        
        # Remove "$\n\n" from all elements in the array
        for d in cleaned_array:
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = value

        # Exclude elements with length greater than 500 to exclude whole image extractions
        cleaned_array_processed = [d for d in cleaned_array if len(d.get('text', '')) <= 900]

        # Call the function to extract text values
        processed_array_for_fuzzy = self.make_fuzzy_processed_array(cleaned_array_processed)
        
        # Define the characters you want to remove
        characters_to_remove = ['[', ']', '{', '}', '(', ')']
        processed_array_for_fuzzy = [element.translate(str.maketrans('', '', ''.join(characters_to_remove)) ) for element in processed_array_for_fuzzy]

        #print(processed_array_for_fuzzy)      
        coordinates = ()
        final_json = {}

        for sentence_key in self.template_json.keys():

            for d in cleaned_array_processed:
                for key, value in d.items():
                    sentence_extract = ""
                    if key == "coordinates":
                        coordinates = ()
                        coordinates = value
                    if key == "text" and coordinates:
                        sentence_extract = value

                    if len(sentence_extract) >= 1:
                        # print("SSSSSS", [sentence_key])
                        # Call the fuzzy search function for the current key
                        
                        if(sentence_key == "Payer's name and address — Nom et adresse du payeur"):
                            output = processed_array_for_fuzzy[0]
                            if output == sentence_key:
                                output = ""
                        elif(sentence_key == "Recipient's name (last name first) and address — Nom, prénom et adresse du bénéficiaire"):
                            output = processed_array_for_fuzzy[1]
                            if output == "Recipient's name last name first and address — Nom, prénom et adresse du bénéficiaire":
                                output = ""
                        elif (sentence_key == "Other Information - Box 1" or sentence_key == "Other Information - Amount 1" or sentence_key == "Other Information - Box 2" or sentence_key == "Other Information - Amount 2" or sentence_key == "Other Information - Box 3" or sentence_key == "Other Information - Amount 3"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "Other Information filtos tenesignemente").replace(" ,", ",").replace(", ", ",")
                        elif(sentence_key == "25 Taxable amount of eligible dividends"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "Taxable amount of eligible dividends")
                        else:
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, sentence_key)
                        
                        if output is not None:
                            if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                                interim_dict = {}

                                output = output.replace("\n\n", "  ").strip()
                                
                                if (sentence_key == "Recipient's name (last name first) and address — Nom, prénom et adresse du bénéficiaire" or sentence_key == "Payer's name and address — Nom et adresse du payeur"):
                                    interim_dict["Value"] = output.replace("\n","").strip()
                                elif (sentence_key == "21 Report Code"):
                                    interim_dict["Value"] = output[:1].strip()
                                elif (sentence_key == "Other Information - Box 1"):
                                    interim_dict["Value"] = output.split()[0]
                                elif (sentence_key == "Other Information - Amount 1"):
                                    interim_dict["Value"] = output.split()[1]
                                elif (sentence_key == "Other Information - Box 2"):
                                    interim_dict["Value"] = output.split()[2]
                                elif (sentence_key == "Other Information - Amount 2"):
                                    interim_dict["Value"] = output.split()[3]
                                elif (sentence_key == "Other Information - Box 3"):
                                    interim_dict["Value"] = output.split()[4]
                                elif (sentence_key == "Other Information - Amount 3"):
                                    interim_dict["Value"] = output.split()[5]
                                else:
                                    interim_dict["Value"] = output.strip().split("  ")[0].strip()
                                
                                interim_dict["Coordinates"] = coordinates
                                final_json[sentence_key] = interim_dict

        final_json = {key.replace("\n", " "):val for key, val in final_json.items()}
        return final_json