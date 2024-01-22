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


class PDF_1098E_ImageExtractor:
    """
    centriod_closeness_threshold=90 #For 1098E
    threshold=0.9 #For 1098
    page_number=1 (Page Number to convert into image)
    """

    def __init__(self, pdf_file_path, output_image_path, template_json, threshold=0.9, page_number=1,
                 centriod_closeness_threshold=90):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.page_number = page_number
        self.centriod_closeness_threshold = centriod_closeness_threshold
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
        print(f"Squares Detection Succesfull with Count: {len(squares)}")
        return squares

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    def are_close(self, centroid1, centroid2):
        return abs(centroid1[0] - centroid2[0]) <= self.centriod_closeness_threshold and abs(
            centroid1[1] - centroid2[1]) <= self.centriod_closeness_threshold

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

        print(f"Final Centroids Count: {len(filtered_centroids)}")
        return filtered_centroids, grouped_squares

    def final_contour_data_extraction(self, grouped_squares):
        start_time = time.time()
        image = cv2.imread(self.output_image_path)
        data = []

        for i, squares_in_group in enumerate(grouped_squares):
            if len(squares_in_group) > 0:
                square = squares_in_group[0]
                x, y, w, h = cv2.boundingRect(square)
                cropped_image = image[y:y + h, x:x + w]
                coord_dict = {}
                coord_dict['coordinates'] = (y, y + h, x, x + w)
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray_image, lang='eng', config='--psm 6')
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
            filtered_centroids, grouped_squares = self.calculate_final_centroid_square_group(squares)

            data = self.final_contour_data_extraction(grouped_squares)
            return data
        else:
            return []

    def search_keyword_and_extract_value(self, array, keyword):
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

    def make_fuzzy_processed_array(self, dictionary_list):
        processed_array_for_fuzzy = []

        for item in dictionary_list:
            text_value = item.get('text', '').strip()  # Get the 'text' value and remove leading/trailing whitespace
            if text_value:  # Check if the text value is not empty
                processed_array_for_fuzzy.append(text_value)

        return processed_array_for_fuzzy

    # Function to extract data from a single PDF file
    def process_single_pdf_1098E(self, pdf_file_path):

        extracted_data = self.process_pdf_to_text()
        cleaned_array = extracted_data.copy()

        # Remove "$\n\n" from all elements in the array
        for d in cleaned_array:
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = value

        # Exclude elements with length greater than 500 to exclude whole image extractions
        cleaned_array_processed = [d for d in cleaned_array if len(d.get('text', '')) <= 500]

        # Call the function to extract text values
        processed_array_for_fuzzy = self.make_fuzzy_processed_array(cleaned_array_processed)

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
                        # Call the fuzzy search function for the current key
                        output = self.search_keyword_and_extract_value(processed_array_for_fuzzy, sentence_key)

                        # If a result is found, update the final_json
                        if output is not None:
                            if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                                interim_dict = {}
                                

                                output = output.replace("|", "").strip()
                                # .split("\n\n")[0] <- Only for 1098E
                                match_factor = self.is_similarity_above_threshold(
                                    "RECIPIENT’S/LENDER’S name, street address, city or town, state or\nprovince, country, ZIP or foreign postal code, and telephone number",
                                    sentence_key)

                                if not match_factor:
                                    interim_dict["Value"] = output.split("\n\n")[0].strip()
                                else:
                                    interim_dict["Value"] = output.replace("\n\n", " ").strip()
                                interim_dict["Coordinates"] = coordinates
                                sentence_key = sentence_key.replace("\n", " ")
                                final_json[sentence_key] = interim_dict

        return final_json