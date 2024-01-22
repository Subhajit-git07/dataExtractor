
import fitz
import re
import numpy as np
import cv2 as cv
import os
import pytesseract
import cv2
import time
# import pandas as pd
import Levenshtein
# import matplotlib.pyplot as plt
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\JZ548KE\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


class PDF_1099_MISC_ImageExtractor:
    """
    centriod_closeness_threshold=0 #For 1098
    threshold=0.9 #For 1098
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path, template_json, threshold=0.9, page_number=1, centriod_closeness_threshold=70):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.page_number = page_number
        self.centriod_closeness_threshold = centriod_closeness_threshold
        self.data_keywords = template_json#{eky:"" for eky, vla in template_json.items()}
        # self.tjson = template_json

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
            print(self.output_image_path)
            print("Outer Bounding Box has been added and Image Generated")
        except Exception as e:
            print(f"Error: Outer Bounding Box generation Failed due to OpenCV error!! {str(e)}")
    

    def thicken_lines(self):
        import copy
        table_img = cv2.imread(self.output_image_path)
        
        result = copy.deepcopy(table_img)

        gray_img = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

        thresh_image = cv2.threshold(gray_img, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#         cv2.imwrite("pdf_1099_misc/thresh_image.png", thresh_image)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        remove_horizontal = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)

        veritical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, veritical_kernel, iterations=3)

        cnts1 = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
        cnts1 = cnts1[0] 

        cnts_v = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_v = cnts_v[0] if len(cnts_v) == 2 else cnts_v[1]

        for c1 in cnts1:
            cv2.drawContours(result,[c1],-1,(0, 0, 0),4)

        for c_v in cnts_v:
            cv2.drawContours(result,[c_v], -1, (0, 0, 0), 4)
            
#         cv2.imwrite(self.output_image_path, result) 
        return 

    
    def find_squares(self):
        img = cv2.imread(self.output_image_path)
                
        img = cv.GaussianBlur(img, (3, 3), 0)
        squares = []
        for gray in cv.split(img):
            for thrs in range(0, 255, 26):
                if thrs == 0:
                    bin = cv.Canny(gray, 0, 50, apertureSize=5)
                    bin = cv.dilate(bin, None)
                else:
                    _, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
                contours, _ = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_len = cv.arcLength(cnt, True)
                    cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
                    
                    area = cv2.contourArea(cnt)
                    hull_area = cv2.contourArea(cv2.convexHull(cnt))
                    solidity = float(area)/hull_area if hull_area != 0 else 0
                    
                    if solidity>=0.95 and len(cnt) == 4 and area > 5000 and cv.isContourConvex(cnt):
#                         print("SOLIDITY", [solidity, hull_area])
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                        if max_cos < self.threshold:
#                             print("MAX COS", [max_cos])
#                             if solidity>=1.0:
                            squares.append(cnt)
#         print(f"Squares Detection Succesfull with Count: {len(squares)}")
        
#         image_dis = cv2.imread(self.output_image_path)
#         cv2.drawContours(image_dis, squares, -1, (0, 255, 0), 1)

# #         Create a larger figure size
#         plt.figure(figsize=(15, 10))  # Adjust the width and height as needed

#         # Show the image using Matplotlib
#         plt.imshow(cv2.cvtColor(image_dis, cv2.COLOR_BGR2RGB))

#         # Add a title (optional)
#         plt.title("Enlarged Image")

#         # Display the image
#         plt.show()
        return squares

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    def are_close(self, centroid1, centroid2):
        return abs(centroid1[0] - centroid2[0]) <= self.centriod_closeness_threshold and abs(centroid1[1] - centroid2[1]) <= self.centriod_closeness_threshold

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_final_centroid_square_group(self, squares):
        centroids = []
        for square in squares:
            M = cv.moments(square)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))

        centroids = list(set(centroids))

        filtered_centroids = [centroids[0]]
        for i in range(1, len(centroids)):
            is_duplicate = False
            for j in range(len(filtered_centroids)):
                if self.are_close(centroids[i], filtered_centroids[j]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_centroids.append(centroids[i])

        filtered_centroids.sort(key=lambda x: (x[0], x[1]))

        grouped_squares = [[] for _ in filtered_centroids]

        for square in squares:
            square_centroid = (
                int((square[0][0] + square[1][0] + square[2][0] + square[3][0]) / 4),
                int((square[0][1] + square[1][1] + square[2][1] + square[3][1]) / 4)
            )

            closest_centroid_idx = min(
                range(len(filtered_centroids)),
                key=lambda i: self.calculate_distance(square_centroid, filtered_centroids[i])
            )

            grouped_squares[closest_centroid_idx].append(square)

#         print(f"Final Centroids Count: {len(filtered_centroids)}")
#         Sort the filtered centroids by X and Y coordinates
#         filtered_centroids.sort(key=lambda x: (x[0], x[1]))

#         # Create a larger figure size
#         plt.figure(figsize=(15, 10))

        # Plot the filtered centroids as red points and print their coordinates
#         for cX, cY in filtered_centroids:
#             plt.scatter(cX, cY, marker='x', c='r')
#             print(f"Centroid Coordinates: ({cX}, {cY})")

# #         print("Path", self.output_image_path)
#         plt.imshow(cv2.imread(self.output_image_path))
#         plt.title("Filtered Centroids of Cells")
#         plt.show()
        return filtered_centroids, grouped_squares

    def final_contour_data_extraction(self, grouped_squares):
#         start_time = time.time()
        image = cv2.imread(self.output_image_path)
        data = []
        data_coord_dict = {}
        for i, squares_in_group in enumerate(grouped_squares):
#             if i!=0:continue
            if len(squares_in_group) > 0:
                square = squares_in_group[0]
                x, y, w, h = cv2.boundingRect(square)
                cropped_image = image[y:y + h, x:x + w]
#                 cv2.imwrite("cropped_images/"+f"{i+1}.png", cropped_image)
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite("cropped_images/"+f"gray_{i+1}.png", gray_image)
                text = pytesseract.image_to_string(gray_image, config="--oem 3  --psm 6", lang='eng')
#                 text = pytesseract.image_to_string(gray_image, lang='eng')
                data_coord_dict = {"text":text, "coords":(y, y + h, x, x + w)}
                data.append(data_coord_dict) 
#         print("Text extraction completed.")
#         end_time = time.time()
#         print(f"Elapsed Time: {end_time - start_time}")
        return data
    
    @staticmethod
    def match_confidence(actual_str, mapping_str):
        float_point_value = Levenshtein.ratio(actual_str, mapping_str)
        confidence = float_point_value * 100
        return confidence
    
    @staticmethod
    def segregate_recipient_name(address_list):
        address_text_str = '\n'.join(address_list)
        #['RECIPIENT’S name', 'fdffsdk ewew', 'Street address (including apt. no.)', 'rewr kelqwe weoqeq', 'City or town, state or province, country, and ZIP or foreign postal code\n3eriower porwerpwoe']
        find_list = ["RECIPIENT'S name", "Street address", "City or town"]
        index_list = []

        address_info_list = address_text_str.split('\n')

        for idx, address_text in enumerate(address_info_list):
            for ftext in find_list:
                if address_text.find(ftext)!=-1:
                    index_list.append(idx)
                    break
                    
        info_dict = {
            address_info_list[index_list[0]]: "",
            address_info_list[index_list[1]]: "",
            }

        if index_list[1]-index_list[0]>1:
            value1 = ' '.join(address_info_list[index_list[0]+1:index_list[1]])
            key1 = address_info_list[index_list[0]]
            info_dict[key1] = value1
        if index_list[2]-index_list[1]>1:
            value2 = ' '.join(address_info_list[index_list[1]+1:index_list[2]])
            key2 = address_info_list[index_list[1]]
            info_dict[key2] = value2

        key3_val_list = address_info_list[index_list[2]:]
        if len(key3_val_list) > 1:
            key3 = key3_val_list[0]
            val3_2 = ' '.join(key3_val_list[1:])
            val3 = ''
            if len(key3.split('\n'))>1:
                key3_list = key3.split('\n')
                val3 = ' '.join(key3_list[1:])
                key3 = key3_list[0]
            val3 = ' '.join([val3, val3_2])
            info_dict[key3] = val3
        else:
            key3 = key3_val_list[0]
            val3 = ''
            if len(key3.split('\n'))>1:
                key3_list = key3.split('\n')
                val3 = ' '.join(key3_list[1:])
                key3 = key3_list[0]
            val3 = ' '.join([val3, val3_2])
            info_dict[key3] = val3
        return info_dict
        
    def mapping_data(self, extracted_data):
        final_list = []
        final_json_dict = {}
        for row_dict in extracted_data:
            try:
                row = row_dict["text"]
                row = row.replace("’", "'")
                row_list = [key_text.strip() for key_text in row.split('\n\n') if key_text.strip()]
                if not row_list:continue
                # print("RRRRR", row_list)
                edata = ' '.join(row_list)
                exe_flag = False
                multi_val_flag = False
                # print("MMMM", row_list[0])
                if len(row_list)>1 and row_list[0].find("RECIPIENT'S name")!=-1:
                    info_dict = self.segregate_recipient_name(row_list)
                    # print("HHHHHHHHHHHHHH")
                    for mkey, mval in info_dict.items():
                        for jkey in self.data_keywords:
                            mkey = mkey.replace('’', "'")
                            if self.match_confidence(jkey, mkey) > 90.00:
                                del self.data_keywords[jkey]
                                self.data_keywords[jkey] = {"Value":mval, "Coordinates":row_dict["coords"]}
                                break

                elif len(row_list)==1:
                    extracted_text = row_list[0].strip()
                    bn_split = extracted_text.split('\n')
                    if bn_split[0].replace('’', "'").find("PAYER'S name")!=-1:
                        key_index = extracted_text.find("telephone no")
                        if key_index!=-1:
                            key_index = key_index+len("telephone no")+1
                            value = extracted_text[key_index:].strip()
                            comp_text = extracted_text[:key_index]
                            exe_flag = True
                    
                    elif extracted_text.replace('’', "'").find("PAYER'S name")!=-1:
                        row_str = ' '.join(row_list)
                        key_index = row_str.find("telephone no")+len("telephone no")
                        if key_index!=-1:
                            comp_text = row_str[:key_index]
                            value = row_str[key_index:]
                            # print("DDDDDDD")
                            # print([comp_text], [value])

                    elif len(bn_split)==2:
                        comp_text, value = bn_split
                        exe_flag = True
    #                     print("COMP", [comp_text, value])
                    elif len(bn_split)>1:
                        if extracted_text.find("16 State tax withheld")!=-1 or                             extracted_text.replace("’", "'").find("17 State/Payer's state no")!=-1 or                                 extracted_text.find("18 State income")!=-1:
                            comp_text, value = extracted_text.rsplit("\n", 1)
                            exe_flag = True
                            multi_val_flag = True
    #                         print("COMP", [comp_text, value])
                        else:
                            comp_text, value = extracted_text.rsplit("\n", 1)
                            exe_flag = True
                    elif len(bn_split)==1:
                        comp_text = bn_split[0]
                        value = ''
                        exe_flag = True
                elif edata.replace('’', "'").find("PAYER'S name")!=-1 and edata.find("calendar year")==-1:
                    key_index = edata.find("telephone no")
                    if key_index!=-1:
                        key_index = key_index+len("telephone no")+1
                        value = edata[key_index:].strip()
                        comp_text = edata[:key_index]
                        exe_flag = True    
                if exe_flag:
                    for map_keyword in self.data_keywords:
                        if self.match_confidence(map_keyword, comp_text) > 85.00 and not multi_val_flag:
                            coords = row_dict["coords"]
                            final_list.append([map_keyword, value])
    #                         map_keyword = map_keyword.replace("\n", " ")
                            self.data_keywords[map_keyword] = {"Value":value, "Coordinates":coords}
                            break
                        elif multi_val_flag:
    #                         print("TEXT", [comp_text, value])
                            temp_list = comp_text.split('\n')
    #                         comp_text_list = comp_text.split('\n')
                            if len(temp_list)==1:
                                comp_text_temp = temp_list[0]
                                if self.match_confidence(map_keyword, comp_text_temp) > 80.00:
                                    coords = row_dict["coords"]
                                    self.data_keywords[map_keyword] = {"Value":value, "Coordinates":coords}
                                    break
                            elif len(temp_list)==2:
                                comp_text_temp, value1 = temp_list
                                if self.match_confidence(map_keyword, comp_text_temp) > 80.00:
                                    coords = row_dict["coords"]
                                    self.data_keywords[map_keyword] = [{"Value":value1, "Coordinates":coords}, {"Value":value, "Coordinates":coords}]
                                    break
            except:
                continue
                
#         df = pd.DataFrame(final_list)
#         display(df)
        # self.data_keywords = sorted(self.data_keywords, key=lambda x:self.tjson[x])
        final_json_dict = {}
        for key, val in self.data_keywords.items():
            key = key.replace('\n', ' ')
            if isinstance(val, list):
                key1 = key+"_1"
                key2 = key+"_2"
                final_json_dict[key1] = val[0]
                final_json_dict[key2] = val[1]
                continue
            if not val:
                val = {"Value":'', "Coordinates":(0, 0, 0, 0)}
            final_json_dict[key] = val
        # self.data_keywords = {key.replace("\n", " "): val for key, val in self.data_keywords.items()}
        return final_json_dict 
        
    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            self.add_outer_bounding_box()
#             self.thicken_lines()
#             self.remove_dotted_lines()
            squares = self.find_squares()
            filtered_centroids, grouped_squares = self.calculate_final_centroid_square_group(squares)
            data = self.final_contour_data_extraction(grouped_squares)
            return data
        return []
    
    def process_single_pdf_1099_misc(self):
        extracted_data = self.process_pdf_to_text()
        final_json_dict = self.mapping_data(extracted_data)
        return final_json_dict

