import fitz
from flask import *
import numpy as np
import cv2 as cv
import os
import math
import pytesseract
import cv2
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class PDF_1099_OID_ImageExtractor:
    """
    centriod_closeness_threshold=70 #For 1099 INT
    threshold=0.9 #For 1099
    page_number=1 (Page Number to convert into image)
    """
    def __init__(self, pdf_file_path, output_image_path, template_json, threshold=0.9,page_number=1,centriod_closeness_threshold=50, threshold_cosine = 0.50):
        self.pdf_file_path = pdf_file_path
        self.output_image_path = output_image_path
        self.threshold = threshold
        self.page_number = page_number
        self.centriod_closeness_threshold = centriod_closeness_threshold
        self.threshold_cosine = threshold_cosine
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
            print(self.output_image_path)
            print("Outer Bounding Box has been added and Image Generated")
        except Exception as e:
            print(f"Error: Outer Bounding Box generation Failed due to OpenCV error!! {str(e)}")

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
                    
                    if solidity==1.0 and len(cnt) == 4 and area > 5000 and cv.isContourConvex(cnt):
#                         print("SOLIDITY", [solidity, hull_area])
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                        if max_cos < self.threshold:
#                             print("MAX COS", [max_cos])
#                             if solidity>=1.0:
                            squares.append(cnt)
#         print(f"Squares Detection Succesfull with Count: {len(squares)}")
        
        image_dis = cv2.imread(self.output_image_path)
        cv2.drawContours(image_dis, squares, -1, (0, 255, 0), 1)

#         Create a larger figure size
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

        print(f"Final Centroids Count: {len(filtered_centroids)}")
        
        return filtered_centroids, grouped_squares

    def final_contour_data_extraction(self, grouped_squares):
        start_time = time.time()
        image = cv2.imread(self.output_image_path)
        data = []
        coord_dict = {}

        for i, squares_in_group in enumerate(grouped_squares):
            if len(squares_in_group) > 0:
                square = squares_in_group[0]
                x, y, w, h = cv2.boundingRect(square)
                cropped_image = image[y:y + h, x:x + w]
                coord_dict = {}
                coord_dict['coordinates'] = (y, y + h, x, x + w)
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(cropped_image, lang='eng', config = "--psm 6")
                coord_dict['text'] = text
                data.append(coord_dict)

        print("Text extraction completed.")
        end_time = time.time()
        print(f"Elapsed Time: {end_time - start_time}")
#         print(data)
        return data

    def process_pdf_to_text(self):
        if self.convert_pdf_to_image():
            self.add_outer_bounding_box()
            squares = self.find_squares()
            filtered_centroids, grouped_squares = self.calculate_final_centroid_square_group(squares)
#             # Create a larger figure size
#             plt.figure(figsize=(15, 10))

#             # Plot the filtered centroids as red points and print their coordinates
#             for cX, cY in filtered_centroids:
#                 plt.scatter(cX, cY, marker='x', c='r')
#                 print(f"Centroid Coordinates: ({cX}, {cY})")
#             image = cv2.imread(self.output_image_path)
#             plt.imshow(image)
#             plt.title("Filtered Centroids of Cells")
#             plt.show()
            
            data = self.final_contour_data_extraction(grouped_squares)
            return data
        else:
            return []
        
    def process_single_pdf_1099_oid(self):
        
        extracted_data = self.process_pdf_to_text()
        cleaned_array = extracted_data.copy()
        # Remove "$\n\n" from all elements in the array
        for d in cleaned_array:
            for key, value in d.items():
                if isinstance(value, str):
                    d[key] = value.replace("$\n\n", "").replace("= ", "").replace('“','').replace('°','').replace('"', ' ').replace('|','').replace('  ',' ')
        # Exclude elements with length greater than 500 to exclude whole image extractions
        cleaned_array_processed = [d for d in cleaned_array if len(d.get('text', '')) <= 500]
        #print(cleaned_array_processed)
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

                        # To checks to make sure we have the correct sentences for checking similarity.
                        process_1 = func_check_similarity(sentence_key, sentence_extract, self.threshold_cosine)
                        process_2 = func_check_similarity(
                            sentence_key[:math.floor(len(sentence_key) / 3)],
                            sentence_extract[:math.floor(len(sentence_key) / 3)], (self.threshold_cosine-0.1))

                        if process_1 and process_2:
                            output = data_extract(sentence_key, sentence_extract)

                            if output is not None:
                                if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                                    interim_dict = {}
                                    interim_dict["Value"] = output
                                    interim_dict["Coordinates"] = coordinates
                                    if sentence_key == 'For calendar year':
                                        interim_dict["Value"] = interim_dict["Value"].replace(" ", "").replace("_","")
                                    if sentence_key == "RECIPIENT'S name":
                                        key_val_dict = seggregate_keys(sentence_extract)
                                        for key, value in key_val_dict.items():
                                            final_json[key] = {"Value":value, "Coordinates":coordinates}
                                    elif sentence_key in ["12 State","13 State identification no.","14 State tax withheld"]:
                                        key1, value1, key2, value2 = handle_multiple_val_fields(sentence_key, sentence_extract)
                                        final_json[key1] = {"Value":value1, "Coordinates":coordinates}
                                        final_json[key2] = {"Value":value2, "Coordinates":coordinates}
                                    else:
                                        final_json[sentence_key] = interim_dict


                        if final_json.get(sentence_key) is None or final_json.get(sentence_key) == "":
                            if sentence_key!= '11 Bond premium' and process_1:
                                output = data_extract(sentence_key, sentence_extract)
                                interim_dict = {}
                                interim_dict["Value"] = output
                                interim_dict["Coordinates"] = coordinates
                                if sentence_key == 'For calendar year':
                                    interim_dict["Value"] = interim_dict["Value"].replace(" ", "").replace("_","")
                                if sentence_key == "RECIPIENT'S name":
                                    key_val_dict = seggregate_keys(sentence_extract)
                                    for key, value in key_val_dict.items():
                                        final_json[key] = {"Value":value, "Coordinates":coordinates}
                                elif sentence_key in ["12 State","13 State identification no.","14 State tax withheld"]:
                                    key1, value1, key2, value2 = handle_multiple_val_fields(sentence_key, sentence_extract)
                                    final_json[key1] = {"Value":value1, "Coordinates":coordinates}
                                    final_json[key2] = {"Value":value2, "Coordinates":coordinates}
                                else:
                                    final_json[sentence_key] = interim_dict

        return final_json

def func_check_similarity(sentence1, sentence2, threshold_value=0.7):
    # Set your similarity threshold (e.g., 0.7 for 70% similarity)
    check_similarity = False
    threshold = threshold_value

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2[:len(sentence1) + 1]])

    # Calculate the cosine similarity between the vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # Compare the cosine similarity score to the threshold
    if cosine_sim >= threshold:
        check_similarity = True
    #         print("confidence_score", cosine_sim[0][0])
    else:
        check_similarity = False

    return check_similarity


def data_extract(sentence1, sentence2):
    # Tokenize the sentences into words
    words_sentence1 = sentence1.split()
    words_sentence2 = sentence2.split()

    # Combine the words from both sentences
    all_words = words_sentence1 + words_sentence2

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the words into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_words)

    # Calculate the cosine similarity between the TF-IDF vectors of all words
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Calculate the length of the shortest sentence
    min_length = min(len(words_sentence1), len(words_sentence2))

    # Initialize a variable to store the end index
    end_index = -1  # Default value if no match is found

    # Iterate through the words and compare cosine similarity
    for i in range(min_length):
        if round(cosine_sim[i][i], 1) == 1.0:
            end_index = i

    # Add 1 to the end index to get the position after the end of the matched part
    end_index += 1

    #     print(f"End Index of Sentence 1 in Sentence 2: {end_index}")

    words = sentence2.split()

    # Specify the word index (0-based) from where you want to extract words
    word_index = end_index  # Change this to the desired word index

    # Extract words from the specified index to the end of the sentence
    words_after_index = words[word_index:]

    # Join the extracted words to form a new sentence
    extracted_sentence = ' '.join(words_after_index)

    return extracted_sentence

def seggregate_keys(extracted_sent):
    
    try:
        sentence = extracted_sent
        print(extracted_sent)
        # Define regular expressions to match the patterns
        recipient_name_pattern = r"RECIPIENT'S name\n\n(.*?)\n|RECIPIENT'S name\n(.*?)\n|RECIPIENT’S name\n\n(.*)\n|RECIPIENT’S name\n(.*)\n"
        street_address_pattern = r"Street address [(]including apt. no.[)]\n\n(.*?)\n|Street address [(]including apt. no.[)]\n(.*?)\n"
        city_state_zip_pattern = r"City or town, state or province, country, and ZIP or foreign postal code\n\n(.*?)\n|City or town, state or province, country, and ZIP or foreign postal code\n(.*?)\n"

        # Use regular expressions to extract the values
        recipient_name = re.search(recipient_name_pattern, sentence).groups()
        street_address = re.search(street_address_pattern, sentence).groups()
        city_state_zip = re.search(city_state_zip_pattern, sentence).groups()

        recipient_name = [x for x in recipient_name if x!=None]
        street_address = [x for x in street_address if x!=None]
        city_state_zip = [x for x in city_state_zip if x!=None]

        seggregate_dict = {"RECIPIENT'S name":recipient_name[0], "Street address (including apt. no.)":street_address[0], "City or town, state or province, country, and ZIP or foreign postal code":city_state_zip[0]}

        return seggregate_dict
    
    except:
        seggregate_dict = {"RECIPIENT'S name":extracted_sent, "Street address (including apt. no.)":'', "City or town, state or province, country, and ZIP or foreign postal code":''}
        return seggregate_dict

def handle_multiple_val_fields(sentence_key, extracted_sent):
    sentence=extracted_sent
    if sentence_key=="14 State tax withheld":
        pattern = r"14 State tax withheld\n(.*)\n"
        output = re.search(pattern, sentence).groups()
        
        key1 = '14(1) State tax withheld'
        key2 = '14(2) State tax withheld'
        try:
            value1 = output[0]
        except:
            value1 = ""
            
        try:
            value2 = output[1]
        except:
            value2 = ""
        
    elif sentence_key=="12 State":
        pattern = r"12 State\n(.*)\n"
        output = re.search(pattern, sentence).groups()
        
        key1 = '12(1) State'
        key2 = '12(2) State'
        try:
            value1 = output[0]
        except:
            value1 = ""
            
        try:
            value2 = output[1]
        except:
            value2 = ""
        
    else:
        pattern = r"13 State identification no.\n(.*)\n"
        output = re.search(pattern, sentence).groups()
        
        key1 = '13(1) State identification no.'
        key2 = '13(2) State identification no.'
        
        try:
            value1 = output[0]
        except:
            value1 = ""
        
        try:
            value2 = output[1]
        except:
            value2 = ""
        
        
    return key1, value1, key2, value2