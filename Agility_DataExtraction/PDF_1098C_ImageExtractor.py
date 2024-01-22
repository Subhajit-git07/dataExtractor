import fitz
# Run OCR on the cropped images

import numpy as np
import cv2 as cv

import os
import pytesseract
import cv2
import time

import re
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

class PDF_1098C_ImageExtractor:
    """
    centriod_closeness_threshold=90 #For 1098C
    threshold=0.9 #For 1098
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path,template_json, threshold=0.9,page_number=1,centriod_closeness_threshold=104):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.page_number = page_number
        self.centriod_closeness_threshold = 104 #centriod_closeness_threshold
        self.template_json = template_json
        self.checkbox_text_list = ["6a Did you provide goods or services in exchange forthe vehicle?",
                    "7 Under the law, the donor may not claim a deduction of more than $500 for this vehicle if thisboxischecked",
                    "6c Describe the goods and services, if any, that were provided. If this box is checked, donee certifies that the goods and services consisted solely of intangible religious benefits",
                    "5b Donee certifies that vehicle is not be transferredto a needy individual for significantly below fair marked value in furtherance of donees charitable purpose",
                    "5a Donee certifies that vehicle will not be transferred for money, other party, or services before completion of material improvements or significant intervening use",
                    "4a Donee certifies that vehicle was sold arms length transaction tounrelated party"]
        
    def calculate_black_pixel_percentage(self,cropped_image, threshold=128):
        # Convert the cropped image to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray_image)

        # Create a binary mask based on the dynamic threshold
        binary_mask = (gray_image <= threshold).astype(np.uint8)

        # Calculate the sum of black pixels and total pixels
        black_pixel_sum = np.sum(binary_mask)
        total_pixels = binary_mask.size

        # Calculate the percentage of black pixels
        black_pixel_percentage = (black_pixel_sum / total_pixels) * 100

        # Determine if the checkbox is checked based on the black pixel density
        checkbox_status = True if black_pixel_percentage >= 10 else False


        return total_pixels,black_pixel_percentage,checkbox_status

    def select_min_order_box(self,checkbox_l, hori,vert):
        #print("inside ",checkbox_l)
        minI = 0
        ln=len(checkbox_l)
        if ln > 1:
            for i in range(ln-1):
                j=i+1
                if hori and checkbox_l[j]['x1']<checkbox_l[minI]['x1']:  
                    minI = j
                if vert and checkbox_l[j]['y1']<checkbox_l[minI]['y1']:  
                    minI = j
        return minI

    def find_order_of_checkbox(self,checkbox_f, checkbox_l):
        ordered_flag=[]
        ver_flag=False
        hor_flag=False
        if len(checkbox_f) == 1:
            ordered_flag.append({'Single Box':True}) if checkbox_f[0] else ordered_flag.append({'Single Box':False})
        elif len(checkbox_f) >= 2:
            ver_flag=True if (abs(checkbox_l[0]['x1'] - checkbox_l[1]['x1'])<5 and (checkbox_l[0]['y1'] != checkbox_l[1]['y1'])) else False
            hor_flag=True if (abs(checkbox_l[0]['y1'] - checkbox_l[1]['y1'])<5 and (checkbox_l[0]['x1'] != checkbox_l[1]['x1'])) else False
            #print('hor_flag   ',hor_flag, " ver_flag ",ver_flag)
            #if hor_flag:
            ln=len(checkbox_l)      
            for i in range(ln):  
                j=self.select_min_order_box(checkbox_l,hor_flag,ver_flag) 
                #print(i, j)
                ordered_flag.append({str(i+1)+" box hori" : checkbox_f[j]}) if hor_flag else ordered_flag.append({str(i+1)+" box vert" : checkbox_f[j]})
                del checkbox_f[j]
                del checkbox_l[j]
                #print(len(checkbox_l))
        return ordered_flag

    def find_checkbox_flag(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squares = []

        checkbox_flag = []
        checkbox_list = []

        # Blur the image and Apply thresholding to enhance contrast
        blur = cv2.GaussianBlur(gray, (1, 1), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #print(contours)
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
    #         if rect[2] < 15 or rect[3] < 15:
    #             continue

            x, y, w, h = rect
            area = w * h

            if area > 300 and area < 1025:

                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):

                     # Crop the image based on the bounding box
                    cropped_image = img[y:y + h, x:x + w]

                    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    avg_intensity = np.mean(gray_image)
                    binary_mask = (gray_image <= 128).astype(np.uint8)
                    black_pixel_sum = np.sum(binary_mask)
                    total_pixels = binary_mask.size
                    black_pixel_percentage = (black_pixel_sum / total_pixels) * 100

                    if black_pixel_percentage <15:
                        squares.append(cnt)
                        #print(black_pixel_percentage)
                        # Calculate the black pixel density
                        total_pixels,black_pixel_percentage,checkbox_status = self.calculate_black_pixel_percentage(cropped_image)
                        checkbox_flag.append(checkbox_status)
                        checkbox_list.append({'x1':x,'x2':x+w,'y1':y,'y2':y+h})
                        #print(f"total_pixels: {total_pixels}, Black Pixel Density: {black_pixel_percentage}, Checkbox Status: {checkbox_status}")

        return checkbox_flag, checkbox_list

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
            
            text = pytesseract.image_to_string(cropped_image, lang='eng',config = "--psm 6")
            ####################### Deriving The Checkbox Logic
            for txt in self.checkbox_text_list:
                if self.is_similarity_above_threshold(text,txt):
                    # Perform the checkbox detection
                    checkbox_flag, checkbox_list= self.find_checkbox_flag(cropped_image)
                    ordered_flag=self.find_order_of_checkbox(checkbox_flag, checkbox_list)
                    ordered_flag_list = [list(flag.values())[0] for flag in ordered_flag]
                    #print(ordered_flag_list)
                    text_add="#Unidentified"
                    if len(ordered_flag)==2:
                        if ordered_flag_list[0]:
                            text_add="#Checked - Yes"
                        elif ordered_flag_list[1]:
                            text_add="#Checked - No"
                        else:
                            text_add="#Unchecked Yes and No"
                    elif len(ordered_flag)==1:
                        if ordered_flag[0]['Single Box']:
                            text_add = "#Checked"
                        else:
                            text_add = "#Unchecked"                       

                    text = f"{text}\n {text_add}"
                    #print(text)
                    
            coord_dict['text'] = text
            data.append(coord_dict)


        print("Text extraction completed.")
        end_time = time.time()
        print(f"Elapsed Time: {end_time - start_time}")
        return data

    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            self.add_outer_bounding_box()
            squares = self.find_squares()
            data = self.final_contour_data_extraction(squares)
            return data
        else:
            return []
        
    def search_keyword_and_extract_value(self, array, keyword):
        try:
            keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            for element in array:
                element = element.replace("'", "’")
                if element.find("Describe the goods and services")!=-1:
                    element = re.sub(r"goods and services[A-Za-z0-9\s\n]+consisted", "goods and services\nconsisted", element)
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
    
    def is_similarity_above_threshold(self,string1, string2, threshold=75):
        similarity_score = fuzz.ratio(string1, string2)
        return similarity_score > threshold

    def make_fuzzy_processed_array(self,dictionary_list):
        processed_array_for_fuzzy = []

        for item in dictionary_list:
            text_value = item.get('text', '').strip()  # Get the 'text' value and remove leading/trailing whitespace
            if text_value:  # Check if the text value is not empty
                processed_array_for_fuzzy.append(text_value)

        return processed_array_for_fuzzy

    # Function to extract data from a single PDF file
    def process_single_pdf_1098C(self,pdf_file_path):
        
        extracted_data = self.process_pdf_to_text()

        # for elem in extracted_data:
            # print("EEEE", elem)

        cleaned_array = extracted_data.copy()
        
        
        # Remove "$\n\n" from all elements in the array
        for d in cleaned_array:
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = value

        # Exclude elements with length greater than 500 to exclude whole image extractions
        cleaned_array_processed = [d for d in cleaned_array if len(d.get('text', '')) <= 900]
        
        #print("cleaned_array_processed",cleaned_array_processed)

        # Call the function to extract text values
        processed_array_for_fuzzy = self.make_fuzzy_processed_array(cleaned_array_processed)

        # for elem in processed_array_for_fuzzy:
            # print("EEEEEEE", [elem])
                
        coordinates = ()
        final_json = {}

        for sentence_key in self.template_json.keys():
            # if sentence_key.find("6c Describe the goods and services, if any, that were provided. If this box is checked, donee certifies that the goods and services\nconsisted solely of intangible religious benefits")==-1:
                # print("KKKKKKKKk")
                # continue

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
                        output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, sentence_key)
                        
                        #print(sentence_key, [output])
                        # If a result is found, update the final_json
                        if output is not None:
                            if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                                interim_dict = {}

                                output = output.replace("\n\n", "").replace("|", "").replace("\n", " ").strip()                                
                                if sentence_key in self.checkbox_text_list:
                                    output=output.split('#')[-1]
                                
                                interim_dict["Value"] = output.strip()
                                interim_dict["Coordinates"] = coordinates
                                final_json[sentence_key] = interim_dict

        final_json = {key.replace("\n", " "):val for key, val in final_json.items()}
        return final_json