import os
import copy
import re

import fitz
import pandas as pd
import numpy as np
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class DigitalFormIdentificationPage:
	def __init__(self, form_page_path):
		self.form_page_path = form_page_path
		self.search_keywords = {

			("T4", "Statement of Remuneration Paid", 
	         "Employee's CPP contributions", "Employee's second QPP contributions",
			 "EI insurable earnings", "CPP/QPP pensionable earnings", 
			 "RPP or DPSP registration number", "PPIP insurable earnings"): "t4",

			("www.irs.gov/Form1099OID", "Bond premium", "Tax-exempt OID",
	         "Early withdrawal penalty"): "1099oid",

			("Revenus d'emploi et revenus divers", "RELEVÉ 1", "Cotisation au RRQ",
	         "Cotisation syndicale", "Pourboires attribués", "Retraite progressive",
			 "Cotisation à un RPA"):"RL1",

			("RELEVÉ 2", "Revenus de retraite et rentes", "Remboursement de primes",
	         "Retrait dans le cadre du RAP", "Prestations d‘un RPA", 
			 "Prestations \(REER, FERR, RPDB", "Autres revenus \(REER ou FERR\)"):"RL2",

			("Statement of Employees Profit Sharing Plan", "Allocations and Payments",
	         "Dividends from Canadian corporations", "T4PS", "Name of employees profit sharing plan",
			 "Dividend tax credit"):"t4ps",

			("Statement of RRSP Income", "T4RSP", "Amounts deemed received", 
	         "Advanced Life Deferred Annuity purchase"):"t4rsp",

			("Statement of Investment Income", "T5", "Protected B",
	         "Dividends from Canadian corporations", "Taxable amount of eligible dividends",
			 "Dividend tax credit for eligible", "Interest from Canadian sources",
			 "Dividend tax credit for dividends"):"t5",

			("Schedule K-3", "Schedule K-3 \(Form 1065\)", "Check to indicate the parts of Schedule K-3 that apply"): "k3",
			("Schedule K-1", "Schedule K-1 \(Form 1065\)"): "K1",

			("Fishing boat proceeds", "www.irs.gov/Form1099MISC",
	         "Crop insurance proceeds",): "1099misc",

			("www.irs.gov/Form1099INT", "Bond premium on Treasury obligations",
	         "Interest on U.S. Savings Bonds and Treasury obligations",
			 "Interest Income"):"1099int",

			("Statement of Pension, Retirement, Annuity",
	         "Pension or superannuation", "Honoraires ou autres sommes",):"t4a",

			("Statement of Trust Income Allocations and Designations", 
	         "Actual amount of eligible dividends", "Montant imposable des dividendes",
			 "Dividend tax credit for eligible dividends", 
			 "Capital gains eligible for deduction"): "t3",

			("Statement of Employment Insurance and Other Benefits", "Taxable tuition assistance",
	         "Quebec income tax", "Non-taxable tuition",): "t4e",

			("RECIPIENT'S/LENDER", "PAYER'S/BORROWER'S TIN", 
	         "Mortgage interest received from payer", "Mortgage origination date",
			 "Points paid on purchase of principal residence", "www.irs.gov/Form1098",
			 "Mortgage insurance",): "1098",
			
			("DONEE'S TIN", "DONOR'S TIN", "Vehicle or other identification number", 
	         "Gross proceeds from sale", "Donee certifies that vehicle was sold in arm",
			 "Odometer mileage", "Date of sale", 
			 "www.irs.gov/Form1098C",): "1098c",

			("RECIPIENT'S TIN", "Total ordinary dividends", "Section 1202 gain",
	         "Section 897 capital gain", "Foreign country or U.S. possession",
			 "Cash liquidation distributions", "Noncash liquidation distributions",
			 "www.irs.gov/Form1099DIV",): "1099div",

			("PAYER'S TIN", "CUSIP number", "Applicable checkbox on Form 8949",
	         "Date sold or disposed", "Accrued market discount", "Wash sale loss disallowed",
			 "Short-term gain or loss", "Profit or \(loss\) realized in", 
			 "Unrealized profit or \(loss\) on", "Cost or other basis", 
			 "www.irs.gov/Form1099B", ): "1099b",

			("BORROWER'S TIN", "BORROWER'S name", "www.irs.gov/Form1098E",
	         "Student loan interest received by lender", 
			 "Check if box 1 does not include loan origination fees",
			 "www.irs.gov/Form1098E", ): "1098e",

			("FILER'S name", "FILER'S employer identification", "STUDENT'S TIN",
	         "Scholarships or grants", "Service Provider/Acct. No",
	         "www.irs.gov/Form1098T"): "1098t",

		}	

	def read_pdf_page(self):
		word_dict = {}
		# writer = pd.ExcelWriter("excel_files/1098E.xlsx", engine="xlsxwriter")
		with fitz.open(self.form_page_path) as document:
			for page_number, page in enumerate(document, 1):
				words = page.get_text("words")
				word_dict = {
								"data":words,
								"rect_width": page.rect.width,
								"rect_height": page.rect.height,
								"mediabox_width": page.mediabox.width,
								"mediabox_height": page.mediabox.height,
							}
				return word_dict
				# df = pd.DataFrame(words)
				# df.to_excel(writer, sheet_name="Sheet" + str(page_number))
		# writer.close()
		return word_dict

	@staticmethod
	def swap_columns(df, col1, col2):
		col_list = list(df.columns)
		x, y = col_list.index(col1), col_list.index(col2)
		col_list[y], col_list[x] = col_list[x], col_list[y]
		df = df[col_list]
		return df
	
	@staticmethod
	def check_alignment_changed(word_dict):
		rect_width = word_dict["rect_width"]
		mediabox_width = word_dict["mediabox_width"]
		rect_height = word_dict["rect_height"]
		mediabox_height = word_dict["mediabox_height"]

		if rect_width != mediabox_width or rect_height != mediabox_height:
			return True
		else:
			return False

	def apply_aligment_logic(self, word_dict):

		alignment_flag  = self.check_alignment_changed(word_dict)

		xl_df = pd.DataFrame(word_dict["data"]).dropna(axis=1, how='all')#.drop(columns=['Unnamed: 0'])
		if alignment_flag:
			xl_df.columns = ["y1", "x2", "y2", "x1", "words", "row", "row_set", "set_words"]

			rect_width = word_dict["rect_width"]

			xl_df = self.swap_columns(xl_df, 'x1', 'x2')
			# print(xl_df)
			xl_df["x2"] = rect_width - xl_df["x2"]

			xl_df["x1"] = rect_width - xl_df["x1"]

			xl_df = self.swap_columns(xl_df, 'x1', 'y1')
			xl_df = self.swap_columns(xl_df, 'x2', 'y2')
			return xl_df
		
		# print("NO ISSUES")
		xl_df.columns = ["x1", "y1", "x2", "y2", "words", "row", "row_set", "set_words"]
		return xl_df
	
	@staticmethod
	def match_confidence(actual_str, mapping_str):
		match_flag = False
		float_point_value = Levenshtein.ratio(actual_str, mapping_str)
		confidence = float_point_value * 100
		if confidence>85.00:
			match_flag = True
		return match_flag
	
	def perform_form_match(self, words_grouped, keywords_tup):
		form_match = False
		match_cnt = 0
		for keyword_text in keywords_tup:
			for sent in words_grouped["words"]:
				sent = sent.replace("’", "'")
				# print(sent)
				# if not self.match_confidence(keyword_text, sent):
					# break				
				if re.search(r""+keyword_text, sent):
					# print("SENT", [keyword_text])
					match_cnt+=1
					break
		# print("MCT", match_cnt)
		if len(keywords_tup) == match_cnt:
			form_match = True
		return form_match

	def process_page(self):
		form_type = "Unidentified"
		try:
			word_dict = self.read_pdf_page()
			page_word_df = self.apply_aligment_logic(word_dict)

			words_grouped = page_word_df.groupby('row')['words'].apply(' '.join).reset_index()

			for keywords_tup, form in self.search_keywords.items():
				if self.perform_form_match(words_grouped, keywords_tup):
					form_type = copy.deepcopy(form)
					break
		except:
			pass
		return form_type

class DigitalDocIdentificationPDF(DigitalFormIdentificationPage):
	def __init__(self, pdf_path):
		self.pdf_path = pdf_path
		self.search_keyword = {

			("Schedule K-3", "Schedule K-3 \(Form 1065\)"): "K3",
			("Schedule K-1", "Schedule K-1 \(Form 1065\)"): "K1",

			("Fishing boat proceeds", "www.irs.gov/Form1099MISC",
	         "Crop insurance proceeds",): "1099_MISC",

			("www.irs.gov/Form1099INT", "Bond premium on Treasury obligations",
	         "Interest on U.S. Savings Bonds and Treasury obligations",
			 "Interest Income"):"1099_INT",

			("Statement of Pension, Retirement, Annuity",
	         "Pension or superannuation", "Honoraires ou autres sommes",):"T4a",


			("Statement of Trust Income Allocations and Designations", 
	         "Actual amount of eligible dividends", "Montant imposable des dividendes",
			 "Dividend tax credit for eligible dividends", 
			 "Capital gains eligible for deduction"): "T3",

			("Statement of Employment Insurance and Other Benefits", "Taxable tuition assistance",
	         "Quebec income tax", "Non-taxable tuition",): "T4e",

			("RECIPIENT'S/LENDER", "PAYER'S/BORROWER'S TIN", 
	         "Mortgage interest received from payer", "Mortgage origination date",
			 "Points paid on purchase of principal residence", "www.irs.gov/Form1098",
			 "Mortgage insurance",): "1098",
			
			("DONEE'S TIN", "DONOR'S TIN", "Vehicle or other identification number", 
	         "Gross proceeds from sale", "Donee certifies that vehicle was sold in arm",
			 "Odometer mileage", "Date of sale", 
			 "www.irs.gov/Form1098C",): "1098C",

			("RECIPIENT'S TIN", "Total ordinary dividends", "Section 1202 gain",
	         "Section 897 capital gain", "Foreign country or U.S. possession",
			 "Cash liquidation distributions", "Noncash liquidation distributions",
			 "www.irs.gov/Form1099DIV",): "1099DIV",

			("PAYER'S TIN", "CUSIP number", "Applicable checkbox on Form 8949",
	         "Date sold or disposed", "Accrued market discount", "Wash sale loss disallowed",
			 "Short-term gain or loss", "Profit or \(loss\) realized in", 
			 "Unrealized profit or \(loss\) on", "Cost or other basis", 
			 "www.irs.gov/Form1099B", ): "1099B",

			("BORROWER'S TIN", "BORROWER'S name", "www.irs.gov/Form1098E",
	         "Student loan interest received by lender", 
			 "Check if box 1 does not include loan origination fees",
			 "www.irs.gov/Form1098E", ): "1098E",

			("FILER'S name", "FILER'S employer identification", "STUDENT'S TIN",
	         "Scholarships or grants", "Service Provider/Acct. No",
	         "www.irs.gov/Form1098T"): "1098T",

		}	

		super().__init__('DigitalDocIdentificationPDF')

	def read_pdf_page(self, digital_page_num_list=[]):
		page_wise_word_dict = {}
		# writer = pd.ExcelWriter("excel_files/1098E.xlsx", engine="xlsxwriter")
		with fitz.open(self.pdf_path) as document:
			for page_number, page in enumerate(document):
				if digital_page_num_list and (page_number not in digital_page_num_list):
					continue
				words = page.get_text("words")
				word_dict = {
								"data":words,
								"rect_width": page.rect.width,
								"rect_height": page.rect.height,
								"mediabox_width": page.mediabox.width,
								"mediabox_height": page.mediabox.height,
							}
				page_wise_word_dict[page_number] = word_dict
		return page_wise_word_dict

	def process_pdf(self, digital_page_list=[]):
		page_wise_word_dict = self.read_pdf_page(digital_page_list)

		digital_classification_dict = {}
		for page_number, word_dict in page_wise_word_dict.items():

			form_type = "Unidentified"
			try:
				page_word_df = self.apply_aligment_logic(word_dict)

				words_grouped = page_word_df.groupby('row')['words'].apply(' '.join).reset_index()

				for keywords_tup, form in self.search_keywords.items():
					if self.perform_form_match(words_grouped, keywords_tup):
						form_type = copy.deepcopy(form)
						break
			except:
				pass
			digital_classification_dict[page_number] = (form_type, 100)
		return digital_classification_dict


if __name__ == '__main__':
	# page_path = "classification_docs/Form 1098-E Digital.pdf"
	page_path = "split_doc_pages/multi_form.pdf"
	# page_path = "split_doc_pages/T4e Fillable.pdf"
	############ Page ##########
	# dfi_obj = DigitalFormIdentificationPage(page_path) ## splited
	# dfi_obj.read_pdf_page() ## splited
	############################

	############ Page PDF ##########
	ddi_pdf = DigitalDocIdentificationPDF(page_path)
	page_class_dict = ddi_pdf.process_pdf()
	print(page_class_dict)


	

