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


class PDF_1099B_ImageExtractor:
    """
    centriod_closeness_threshold=90 #For 1098
    threshold=0.9 #For 1098
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path, template_json, threshold=0.9, page_number=1, centriod_closeness_threshold=50):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.page_number = page_number
        self.centriod_closeness_threshold = centriod_closeness_threshold
        self.data_keywords = template_json

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

    def find_squares(self):
        img = cv2.imread(self.output_image_path)
        img = cv.GaussianBlur(img, (5, 5), 0)
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
                    if len(cnt) == 4 and cv.contourArea(cnt) > 5000 and cv.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                        if max_cos < self.threshold:
                            squares.append(cnt)
#         print(f"Squares Detection Succesfull with Count: {len(squares)}")
        
#         image_dis = cv2.imread(self.output_image_path)
#         cv2.drawContours(image_dis, squares, -1, (0, 255, 0), 1)

        # Create a larger figure size
#         plt.figure(figsize=(15, 10))  # Adjust the width and height as needed

        # Show the image using Matplotlib
#         plt.imshow(cv2.cvtColor(image_dis, cv2.COLOR_BGR2RGB))

        # Add a title (optional)
#         plt.title("Enlarged Image")

        # Display the image
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
        # Sort the filtered centroids by X and Y coordinates
#         filtered_centroids.sort(key=lambda x: (x[0], x[1]))

        # Create a larger figure size
#         plt.figure(figsize=(15, 10))

        # Plot the filtered centroids as red points and print their coordinates
#         for cX, cY in filtered_centroids:
#             plt.scatter(cX, cY, marker='x', c='r')
#             print(f"Centroid Coordinates: ({cX}, {cY})")

#         plt.imshow(self.output_image_path)
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
                text = pytesseract.image_to_string(gray_image, config="--oem 3 --psm 6", lang='eng')
#                 text = pytesseract.image_to_string(gray_image, lang='eng')
                data_coord_dict = {"text": text, "coords":(y, y + h, x, x + w)}
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
    
    def mapping_data(self, extracted_data):
        cnt = 0
        final_list = []
        final_json_dict = {}
        for row_dict in extracted_data:
            row = row_dict["text"]
            row_list = [key_text.strip() for key_text in row.split('\n\n') if key_text.strip()]
            if not row_list:continue
            exe_flag = False
            value2 = ""
            if len(row_list)>1:
                comp_text = row_list[0]
                value = ' '.join(row_list[1:])
                exe_flag = True
            elif row_list[0].lower().find("state tax withheld")!=-1 or                        row_list[0].lower().find("state identification no")!=-1 or                        row_list[0].lower().find("state name")!=-1:
                comp_text_list = row_list[0].strip().split('\n')
#                 print("COMP", comp_text_list)
                if len(comp_text_list)==3:
                    comp_text, value, value2 = comp_text_list
                    value2 = re.sub(r'\$$', '', value2)
                    exe_flag = True
                elif len(comp_text_list)==2:
                    comp_text, value = comp_text_list
                    exe_flag = True 
            else:
                comp_text_list = row_list[0].strip().rsplit('\n', 1)
                if len(comp_text_list)==2:
                    comp_text, value = comp_text_list
                    exe_flag = True
                else:
                    comp_text, value = row_list[0], ''
                    exe_flag = True
            if exe_flag:
                for map_keyword in self.data_keywords:
                    if comp_text.find("Cost or other basis")!=-1:
                        coords = row_dict["coords"]
#                         final_list.append(["1e Cost or other basis", value])
#                         final_json_dict["1e Cost or other basis"] = {"Coordinates":coords, "Value":value}
                        self.data_keywords["1e Cost or other basis"] = {"Value":value, "Coordinates":coords}
                        break
                    elif self.match_confidence(map_keyword, comp_text) > 90.00 and map_keyword.find("Unrealized profit or (loss) on\nopen")==1: 
                        coords = row_dict["coords"]
                        self.data_keywords[map_keyword] = {"Value":value, "Coordinates":coords}
                        break
                    elif self.match_confidence(map_keyword, comp_text) > 95.00:
                        coords = row_dict["coords"]
                        if value and not value2:
                            self.data_keywords[map_keyword] = {"Value":value, "Coordinates":coords}
                        elif value and value2:
                            # map_keyword1 = map_keyword+'_1'
                            # map_keyword2 = map_keyword+'_2'
                            self.data_keywords[map_keyword] = [{"Value":value, "Coordinates":coords}, {"Value":value2, "Coordinates":coords}]
                            # self.data_keywords[map_keyword1] = {"Coordinates":coords, "Value":value}
                            # self.data_keywords[map_keyword2] = {"Coordinates":coords, "Value":value2}
                        elif not value:
                            self.data_keywords[map_keyword] = {"Value":value, "Coordinates":coords}
                        break
                    elif self.match_confidence(map_keyword, comp_text) > 95.00:
                        coords = row_dict["coords"]
                        self.data_keywords[map_keyword] = {"Coordinates":coords, "Value":value}
                        break
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
            squares = self.find_squares()
            filtered_centroids, grouped_squares = self.calculate_final_centroid_square_group(squares)
            data = self.final_contour_data_extraction(grouped_squares)
            return data
        return []
    
    def process_single_pdf_1099B(self):
        extracted_data = self.process_pdf_to_text()
        final_json_dict = self.mapping_data(extracted_data)
        return final_json_dict

