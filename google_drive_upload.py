from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import os
import pickle
import tempfile

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_file_to_drive(filepath, filename):
    creds = None
    # Use the system's temporary directory to store the user's access and refresh tokens
    # temp_dir = tempfile.gettempdir()
    # token_path = os.path.join(temp_dir, 'token.pickle')

    # # The file token.pickle stores the user's access and refresh tokens, and is
    # # created automatically when the authorization flow completes for the first time.
    # if os.path.exists(token_path):
    #     with open(token_path, 'rb') as token:
    #         creds = pickle.load(token)
    # else:
    #     print('No Pickle Found')

    # # If there are no (valid) credentials available, let the user log in.
    # if not creds or not creds.valid:
    #     if creds and creds.expired and creds.refresh_token:
    #         creds.refresh(Request())
    #     else:
    #         flow = InstalledAppFlow.from_client_secrets_file(
    #             'credentials.json', SCOPES)
    #         creds = flow.run_local_server(port=53458)
    #     # Save the credentials for the next run in the temporary directory
    #     with open(token_path, 'wb') as token:
    #         pickle.dump(creds, token)

    service_account_file = 'ricequalitycheck-b6ab3b833e4d.json'  # Path to your service account key file
    # git commit
    creds = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)

    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': filename,
                     'parents': ['1ErSxdy8UtpDoXnwDkHYoeQRSYWCKRR1g']}  # Update with your folder ID if necessary
    media = MediaFileUpload(filepath, mimetype='application/octet-stream')
    file = service.files().create(body=file_metadata,
                                  media_body=media,
                                  fields='id').execute()

    print(f'File ID: {file.get("id")}')
    fileid = file.get("id")
    sharefile = f"https://drive.google.com/file/d/{fileid}/view?usp=sharing"
    
    print(file)
    print(sharefile)

    return fileid
    # service = build('drive', 'v3', credentials=creds)

    # file_metadata = {'name': filename,
    #                  'parents': ['1ErSxdy8UtpDoXnwDkHYoeQRSYWCKRR1g']}
    # media = MediaFileUpload(filepath, mimetype='application/octet-stream')
    # file = service.files().create(body=file_metadata,
    #                               media_body=media,
    #                               fields='id').execute()
    # print(f'File ID: {file.get("id")}')