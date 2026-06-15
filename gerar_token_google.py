"""
Execute este script UMA VEZ no seu Mac para gerar o refresh_token do Google.
O token gerado vai para as variáveis de ambiente do Railway.

Como usar:
  pip install google-auth-oauthlib
  python gerar_token_google.py
"""

from google_auth_oauthlib.flow import InstalledAppFlow
import json, os

CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "<cole seu client_id aqui>")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "<cole seu client_secret aqui>")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]

client_config = {
    "installed": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}

flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

print("\n" + "="*60)
print("✅ TOKEN GERADO COM SUCESSO!")
print("="*60)
print(f"\nGOOGLE_REFRESH_TOKEN={creds.refresh_token}")
print("\nCopie a linha acima e adicione como variável de ambiente no Railway.")
print("="*60 + "\n")
