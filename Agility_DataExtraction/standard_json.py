
def sub_dict_append(output_json):
    new_dict_to_append = {"Confidence": "100", "Span": {}}
    for key, inner_dict in output_json.items():
        length_of_value = 0
        value = inner_dict.get("Value", "")
        length_of_value = len(value)
        new_dict_to_append["Span"] = {"Offset": 0, "Length": length_of_value}
        inner_dict.update(new_dict_to_append)

    return output_json


def append_upper_dict(output_json, append_json, form_type):
    try:
        append_json['DataExtraction']['Pages'][0]['KeyValues'] = output_json
        append_json['Status'] = "Completed"
        append_json['DataExtraction']['FormType'] = form_type
    except:
        append_json['DataExtraction']['Pages'][0]['KeyValues'] = {}
        append_json['Status'] = "Failed"
        append_json['DataExtraction']['FormType'] = form_type

    return append_json


def append_upper_dict_new(output_json, append_json):
    try:
        append_json['Pages'][0]['KeyValues'] = output_json
    except:
        append_json['Pages'][0]['KeyValues'] = {}

    return append_json
