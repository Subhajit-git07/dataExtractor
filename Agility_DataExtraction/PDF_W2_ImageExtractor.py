import fitz
# Run OCR on the cropped images

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


class PDF_W2_ImageExtractor:
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
            if area > 25000 and area < 1000000:
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
            text = pytesseract.image_to_string(cropped_image, lang='eng',config = "--psm 6")
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
    def process_single_pdf_W2(self):
        
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
        
        coordinates = ()
        final_json = {}
        
        name_length = 0
        first_name_len = 0
        emp_name = ""
        
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
                        
                        if(sentence_key == "12a See instructions for box 12-Code" or sentence_key == "12a See instructions for box 12-Amount"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "12a See instructions for box 12")
                        elif (sentence_key == "12d-Code" or sentence_key == "12d-Amount"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "12d")
                        elif (sentence_key == "12c-Code" or sentence_key == "12c-Amount"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "12c")
                        elif (sentence_key == "12b-Code" or sentence_key == "12b-Amount"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "12b")
                        elif (sentence_key == "e Employee's first name and initial" or sentence_key == "f Employee's address and ZIP code" or sentence_key == "e Employee's Suff" or sentence_key == "e Employee's Last Name"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "e Employee's first name and initial Last name Suff.")
                        elif(sentence_key == "15(1) State Employer's state ID number" or sentence_key == "15(2) State Employer's state ID number"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "15 State Employer's state ID number")
                        elif(sentence_key == "16(1) State wages, tips, etc." or sentence_key == "16(2) State wages, tips, etc."):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "16 State wages, tips, etc.")
                        elif(sentence_key == "17(1) State income tax" or sentence_key == "17(2) State income tax"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "17 State income tax")
                        elif(sentence_key == "18(1) Local wages, tips, etc." or sentence_key == "18(2) Local wages, tips, etc."):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "18 Local wages, tips, etc.")
                        elif(sentence_key == "19(1) Local income tax" or sentence_key == "19(2) Local income tax"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "19 Local income tax")
                        elif(sentence_key == "20(1) Locality name" or sentence_key == "20(2) Locality name"):
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, "20 Locality name")
    
                        else:
                            output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, sentence_key)
                        
                        if output is not None:
                            if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                                interim_dict = {}

                                output = output.replace("\n\n", "  ").strip()
                                
                                
                                #For the Code & Amount Segregation
                                if "-Code" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.strip().split("|")[0].strip()[-1].replace('\n', '')
                                    except:
                                        interim_dict["Value"] = output.strip()
                                elif "-Amount" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.strip().split("|")[1].strip()
                                    except:
                                        interim_dict["Value"] = output.strip()
                                
                                #For OMB No Fix
                                elif "OMB No." in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.strip().split(" ")[0].replace('\n', '')
                                    except:
                                        interim_dict["Value"] = output.strip()
                                elif "11 Nonqualified plans" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.strip().split("|")[1].strip().replace('\n', '')
                                    except:
                                        interim_dict["Value"] = output.strip().replace('\n', '')
                                
                                ## Adjusting The Employee First name & Last name
                                elif "e Employee's first name and initial" in sentence_key:
                                    try:
                                        interim_dict["Value"] = ' '.join(output.split("\n")[0].strip().split(" ")[:-2]).replace('\n', '')
                                        emp_name = output.split("\n")[0].strip()
                                        name_length = len(output.split("\n")[0].strip())
                                    except:
                                        interim_dict["Value"] = output.strip().replace("\n"," ")
                                        emp_name = output.split("\n")[0].strip()
                                        first_name_len = len(extract_first_name(output.split("\n")[0].strip()))
                                        name_length = len(output.split("\n")[0].strip())
                                        
                                elif "e Employee's Last Name" in sentence_key:
                                    try:
                                        interim_dict["Value"] = emp_name.split(" ")[-2].replace('\n', '')
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "f Employee's address and ZIP code" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.split("  ")[0][name_length:].replace("\n"," ").strip().replace('\n', '')
                                    except:
                                        interim_dict["Value"] = output.replace("\n"," ").strip()
                                elif "e Employee's Suff" in sentence_key:
                                    try:
                                        interim_dict["Value"] = emp_name.split(" ")[-1].strip().replace('\n', '')
                                    except:
                                        interim_dict["Value"] = emp_name
                                elif "15(1) State Employer's state ID number" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[0]
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "15(2) State Employer's state ID number" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[1]
                                    except:
                                        interim_dict["Value"] = ""
                                elif "16(1) State wages, tips, etc." in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[0]
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "16(2) State wages, tips, etc." in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[1]
                                    except:
                                        interim_dict["Value"] = ""
                                elif "17(1) State income tax" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[0]
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "17(2) State income tax" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[1]
                                    except:
                                        interim_dict["Value"] = ""
                                elif "18(1) Local wages, tips, etc." in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[0]
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "18(2) Local wages, tips, etc." in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[1]
                                    except:
                                        interim_dict["Value"] = ""
                                elif "19(1) Local income tax" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[0]
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "19(2) Local income tax" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[1]
                                    except:
                                        interim_dict["Value"] = ""
                                elif "20(1) Locality name" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[0]
                                    except:
                                        interim_dict["Value"] = ""
                                        
                                elif "20(2) Locality name" in sentence_key:
                                    try:
                                        interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").split("\n")[1]
                                    except:
                                        interim_dict["Value"] = ""
                                else:
                                    interim_dict["Value"] = output.replace(",","").replace("{","").replace("}","").replace(")","").replace('\n', '').strip()
                                    
                                
                                interim_dict["Coordinates"] = coordinates
                                final_json[sentence_key] = interim_dict

        final_json = {key.replace("\n", " "):val for key, val in final_json.items()}
        return final_json