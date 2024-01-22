import os

import requests
import json


class BaseConfigDev():
    token_url = "https://login.microsoftonline.com/5b973f99-77df-4beb-b27d-aa0c70b8482c/oauth2/v2.0/token"
    dh_url = "https://harvesterdev1.ey.com/api/v1/transactions/unstructuredTransactionStatus"
    data = {
        "grant_type": "client_credentials",
        "client_id": "d9d66641-e924-4122-9fd0-d482ace3618d",
        "scope": "api://5d05e366-dfc5-43d3-ac97-68af928a4c2e/.default",
        "client_secret": "T5p8Q~5no4sclDGLa9aeDwyWNeiD2H5YQjHJQa.z"
    }



class BaseConfigUAT():
    token_url = "https://login.microsoftonline.com/5b973f99-77df-4beb-b27d-aa0c70b8482c/oauth2/v2.0/token"
    dh_url = "https://dh-uat.ey.com/api/v1/transactions/unstructuredTransactionStatus"
    data = {
        "grant_type": "client_credentials",
        "client_id": "bd4b9e51-7b1a-4d40-8b13-62afebb1c072",
        "scope": "api://105e2b24-7e2e-4f60-a0df-1f28e8ed3c77/.default",
        "client_secret": "rnf8Q~SRAElX7zeSICq91vWX5DTjf461KFVL0bVp"
    }



class BaseConfigPROD():
    token_url = "https://login.microsoftonline.com/5b973f99-77df-4beb-b27d-aa0c70b8482c/oauth2/v2.0/token"
    dh_url = "https://dh.ey.com/api/v1/transactions/unstructuredTransactionStatus"
    data = {
       "grant_type": "client_credentials",
       "client_id": "860e5524-f19b-4066-82c9-3d999eca1e51",
       "scope": "api://84303e53-0ecb-41a1-ab4e-31f3812e8c3e/.default",
       "client_secret": "l_j8Q~twOOagPVwxT.Z9T_BU6H7a~gBTDGU9QcO."
   }

# End point format
output_format = {
    "transactionId": "",
    "completedDocumentIds": [],
    "failedDocs": [],
    "pendingDocumentIds": [0]
}


# Function to check and update the completed docs and failed docs


def create_put_response(input_param, final_output):
    completed_docs = []
    failed_format = []
    transaction_id = input_param['TransactionId']
    if len(final_output)>=1:
        for d in final_output:
            try:
                id = d['DocumentId']
                output = d['ExtractedData']
            except:
                id = d['DocumentId']
                output = {}
            if id:
                try:
                    if len(output) >= 1:
                        completed_docs.append(id)
                    else:
                        failed_format.append({"failedDocId": id, "failureReason": "Unable to extract data."})
                except:
                    failed_format.append({"failedDocId": id, "failureReason": "Unable to extract data."})


    if len(completed_docs) == 0: completed_docs = [0]

    output_format.update(completedDocumentIds=completed_docs)
    output_format.update(failedDocs=failed_format)
    output_format.update(transactionId=transaction_id)

    return output_format


def msft_end_point_get(token, data):
    files_list = {
        'grant_type': (None, data['grant_type']),
        'client_id': (None, data['client_id']),
        'scope': (None, data['scope']),
        'client_secret': (None, data['client_secret']),
    }
    response = requests.request("GET", token, headers={}, files=files_list)

    try:
        response_dict = response.json()
        authorization = response_dict.get("token_type") + " " + response_dict.get("access_token")
    except Exception as e:
        authorization = ""
        print(f'Unable to get authorization {e}')

    return authorization


def dh_end_point_put(dh_url, authorization, end_response):
    put_response = 0
    if authorization[:7] == "Bearer ":
        payload = json.dumps(end_response)
        headers = {"content-type": "application/json", "Authorization": authorization}
        try:
            status = requests.request("PUT", dh_url, data=payload, headers=headers, verify = False)
            put_response = status.status_code
        except Exception as e:
            print(f'PUT failed {e}')
            put_response = 0

    return put_response


def run_api_logic(config, input_param, final_output):
    try:
        put_response = create_put_response(input_param, final_output)
        print("put_response", put_response)
        token = msft_end_point_get(config.token_url, config.data)
        status = dh_end_point_put(config.dh_url, token, put_response)
    except:
        status = 400
    return status

