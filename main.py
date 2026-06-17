"""
RAN Assistente — Bot Telegram 24/7
Deploy: Render (gratuito) via polling + health check HTTP
"""

import os, json, logging, asyncio
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN     = os.environ["BOT_TOKEN"]
CHAT_ID       = int(os.environ["CHAT_ID"])
GROQ_KEY      = os.environ["GROQ_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_KEY"]

GOOGLE_CLIENT_ID     = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
GOOGLE_REFRESH_TOKEN = os.environ["GOOGLE_REFRESH_TOKEN"]

PENDENTES_FILE = "/tmp/email_pendentes.json"
PNCP_BASE      = "https://pncp.gov.br/api/consulta/v1/contratacoes/publicacao"

# ── Health check HTTP (mantém Render acordado) ─────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"RAN Bot OK")
    def log_message(self, *args):
        pass

def start_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"Health check rodando na porta {port}")

# ── Google Auth ────────────────────────────────────────────────────────────────
def get_google_creds():
    creds = Credentials(
        token=None,
        refresh_token=GOOGLE_REFRESH_TOKEN,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=[
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/tasks",
        ]
    )
    creds.refresh(Request())
    return creds

def gmail_service():    return build("gmail",    "v1", credentials=get_google_creds())
def calendar_service(): return build("calendar", "v3", credentials=get_google_creds())
def tasks_service():    return build("tasks",    "v1", credentials=get_google_creds())

# ── Telegram helper ────────────────────────────────────────────────────────────
async def send(text: str, bot):
    await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="Markdown")

# ── Groq Whisper ───────────────────────────────────────────────────────────────
async def transcribe_voice(tg_file) -> str:
    path = "/tmp/voice.ogg"
    await tg_file.download_to_drive(path)
    with open(path, "rb") as f:
        audio_bytes = f.read()
    logger.info(f"Audio size: {len(audio_bytes)} bytes")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("voice.ogg", audio_bytes, "audio/ogg; codecs=opus")},
            data={"model": "whisper-large-v3", "language": "pt"},
        )
        if r.status_code != 200:
            logger.error(f"Groq error body: {r.text}")
            raise Exception(f"Groq {r.status_code}: {r.json().get('error', {}).get('message', r.text)}")
        return r.json().get("text", "")

# ── Pré-processar datas relativas ──────────────────────────────────────────────
def preprocessar_datas(text: str) -> str:
    from datetime import date
    import re
    hoje = date.today()
    amanha = hoje + timedelta(days=1)
    dias_semana = ["segunda","terça","quarta","quinta","sexta","sábado","domingo"]
    text2 = text.lower()
    text2 = re.sub(r'\bhoje\b', hoje.strftime("%d/%m/%Y"), text2)
    text2 = re.sub(r'\bamanhã\b', amanha.strftime("%d/%m/%Y"), text2)
    for i, dia in enumerate(dias_semana):
        if dia in text2:
            dias_ate = (i - hoje.weekday()) % 7 or 7
            data_dia = hoje + timedelta(days=dias_ate)
            text2 = text2.replace(dia, data_dia.strftime("%d/%m/%Y"))
    return text2

# ── Groq — classificar intenção ────────────────────────────────────────────────
async def classify_intent(text: str) -> dict:
    hoje = datetime.now().strftime("%Y-%m-%d")
    text = preprocessar_datas(text)
    system = f"""Classifica mensagens em português do Ryan Bereta.
Data de hoje: {hoje}. Fuso: America/Sao_Paulo.

Responda APENAS JSON válido, sem markdown:
{{
  "tipo": "TAREFA"|"EVENTO"|"EMAIL"|"ATAS_ES"|"EMAIL_CMD"|"DESCONHECIDO",
  "titulo": "título limpo",
  "data": "YYYY-MM-DD ou null",
  "hora": "HH:MM ou null",
  "destinatario": "email ou null",
  "assunto": "assunto ou null",
  "corpo": "corpo do email ou null",
  "cmd_num": numero_ou_null,
  "cmd_acao": "ok"|"muda"|"ignora"|null,
  "cmd_instrucao": "instrução ou null"
}}

Regras:
- EVENTO: tem data/hora/dia semana, ou palavras: reunião, meeting, call, almoço, jantar, visita, consulta, compromisso
- titulo: sempre inclua nomes de pessoas e detalhes relevantes. Ex: "Reunião com Nil", "Almoço com João", "Consulta médica"
- TAREFA: verbos de ação sem data específica: ligar, pagar, comprar, enviar, fazer, verificar, lembrar
- EMAIL: "envia email", "manda email", "escreve email" para alguém
- ATAS_ES: "atas es", "atas espírito santo", "atas estadual", "atas federal", "atas consórcio"
- EMAIL_CMD: começa com "ok N", "envia N", "muda N:", "ignora N"
- Datas relativas: amanhã=+1 dia, dias da semana=próxima ocorrência futura"""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
            },
        )
        resp = r.json()
        content = resp["choices"][0]["message"]["content"].strip()
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise Exception(f"JSON não encontrado na resposta: {content[:200]}")
        return json.loads(match.group())

# ── Ações Google ───────────────────────────────────────────────────────────────
def criar_tarefa(titulo: str, data: str = None) -> None:
    svc = tasks_service()
    listas = svc.tasklists().list().execute().get("items", [])
    lista_id = next((l["id"] for l in listas if "ASSISTENTE" in l["title"].upper()), listas[0]["id"])
    body = {"title": titulo}
    if data:
        body["due"] = f"{data}T00:00:00Z"
    svc.tasks().insert(tasklist=lista_id, body=body).execute()

def criar_evento(titulo: str, data: str, hora: str = None) -> None:
    svc = calendar_service()
    if hora:
        start_dt = f"{data}T{hora}:00"
        end_obj  = datetime.strptime(f"{data} {hora}", "%Y-%m-%d %H:%M") + timedelta(minutes=30)
        end_dt   = end_obj.strftime("%Y-%m-%dT%H:%M:00")
        start = {"dateTime": start_dt, "timeZone": "America/Sao_Paulo"}
        end   = {"dateTime": end_dt,   "timeZone": "America/Sao_Paulo"}
    else:
        start = end = {"date": data}
    svc.events().insert(calendarId="primary", body={"summary": titulo, "start": start, "end": end}).execute()

def criar_rascunho(para: str, assunto: str, corpo: str) -> None:
    import base64
    from email.mime.text import MIMEText
    svc = gmail_service()
    msg = MIMEText(corpo, "plain", "utf-8")
    msg["to"] = para
    msg["subject"] = assunto
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()

# ── Busca PNCP ES ───────────────────────────────────────────────────────────────
async def buscar_atas_es() -> str:
    hoje = datetime.now().strftime("%Y%m%d")
    res  = {"estadual": [], "federal": [], "consorcio": []}
    CKW  = ["consorcio", "consórcio", "cim", "polinorte"]
    async with httpx.AsyncClient(timeout=30) as client:
        for pagina in range(1, 30):
            try:
                r = await client.get(PNCP_BASE, params={
                    "dataInicial": "20260101", "dataFinal": hoje,
                    "pagina": pagina, "tamanhoPagina": 10,
                    "codigoModalidadeContratacao": 6, "uf": "ES",
                })
                items = r.json().get("data", [])
            except Exception:
                break
            if not items:
                break
            for item in items:
                if not item.get("srp"):
                    continue
                esfera = item.get("orgaoEntidade", {}).get("esferaId", "")
                orgao  = item.get("orgaoEntidade", {}).get("razaoSocial", "")
                objeto = item.get("objetoCompra", "")
                cnpj   = item.get("orgaoEntidade", {}).get("cnpj", "")
                ano    = item.get("anoCompra", "")
                seq    = item.get("sequencialCompra", "")
                entry  = {"orgao": orgao[:50], "objeto": objeto[:80],
                          "valor": item.get("valorTotalEstimado", 0),
                          "link": f"https://pncp.gov.br/app/editais/{cnpj}/{ano}/{seq}"}
                if esfera == "E":
                    res["estadual"].append(entry)
                elif esfera == "F":
                    res["federal"].append(entry)
                elif esfera in ["N","M"] and any(k in orgao.lower() for k in CKW):
                    res["consorcio"].append(entry)
    total = sum(len(v) for v in res.values())
    if not total:
        return "❌ Nenhuma ata SRP encontrada no ES."
    linhas = ["📋 *ATAS PNCP — Espírito Santo 2026*\n"]
    for tipo, emoji in [("estadual","🏛️ ESTADUAIS"),("federal","🇧🇷 FEDERAIS"),("consorcio","🤝 CONSÓRCIOS")]:
        lista = res[tipo]
        if not lista: continue
        linhas.append(f"\n*{emoji} ({len(lista)})*")
        for i, item in enumerate(lista[:5], 1):
            val = f"R$ {item['valor']:,.0f}".replace(",",".") if item["valor"] else "Valor n/d"
            linhas += [f"\n{i}. {item['orgao']}", f"📌 {item['objeto']}", f"💰 {val}", f"🔗 {item['link']}"]
        if len(lista) > 5:
            linhas.append(f"_(+{len(lista)-5} atas)_")
    return "\n".join(linhas)

# ── Handler principal ───────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or update.effective_chat.id != CHAT_ID:
        return

    bot   = context.bot
    texto = ""
    logger.info(f"Mensagem recebida: voz={bool(update.message.voice)} texto={bool(update.message.text)}")

    if update.message.voice:
        try:
            tg_file = await bot.get_file(update.message.voice.file_id)
            texto = await transcribe_voice(tg_file)
            if not texto:
                await send("⚠️ Não consegui transcrever. Tente por texto.", bot)
                return
            await send(f'🎤 _"{texto}"_', bot)
        except Exception as e:
            await send(f"⚠️ Erro no áudio: {e}", bot)
            return
    elif update.message.text:
        texto = update.message.text.strip()
    else:
        return

    try:
        intent = await classify_intent(texto)
    except Exception as e:
        await send(f"⚠️ Erro ao processar: {e}", bot)
        return

    tipo = intent.get("tipo", "DESCONHECIDO")

    if tipo == "ATAS_ES":
        await send("🔍 Buscando atas no PNCP...", bot)
        await send(await buscar_atas_es(), bot)

    elif tipo == "EMAIL":
        try:
            criar_rascunho(intent.get("destinatario",""), intent.get("assunto","Sem assunto"), intent.get("corpo",""))
            await send(f"📧 *Email pronto!*\nPara: {intent.get('destinatario','')}\nAssunto: {intent.get('assunto','')}\n\n👆 Abra e envie:\nhttps://mail.google.com/mail/u/0/#drafts", bot)
        except Exception as e:
            await send(f"⚠️ Erro ao criar rascunho: {e}", bot)

    elif tipo == "TAREFA":
        try:
            criar_tarefa(intent["titulo"], intent.get("data"))
            msg = f"✅ *Tarefa criada:* {intent['titulo']}"
            if intent.get("data"): msg += f"\n📅 Vence: {intent['data']}"
            await send(msg, bot)
        except Exception as e:
            await send(f"⚠️ Erro ao criar tarefa: {e}", bot)

    elif tipo == "EVENTO":
        if not intent.get("data"):
            await send("📅 Qual é a data do compromisso?", bot)
            return
        try:
            criar_evento(intent["titulo"], intent["data"], intent.get("hora"))
            hora_str = f" às {intent['hora']}" if intent.get("hora") else ""
            await send(f"📅 *Evento criado:* {intent['titulo']}\n🗓️ {intent['data']}{hora_str}", bot)
        except Exception as e:
            await send(f"⚠️ Erro ao criar evento: {e}", bot)

    elif tipo == "EMAIL_CMD":
        try:
            with open(PENDENTES_FILE) as f:
                pendentes = json.load(f)
        except Exception:
            pendentes = []
        num  = intent.get("cmd_num")
        acao = intent.get("cmd_acao")
        if not num or num > len(pendentes):
            await send(f"⚠️ Email [{num}] não encontrado.", bot)
            return
        item = pendentes[num-1]
        if acao == "ignora":
            item["status"] = "ignorado"
            await send(f"🗑️ Email [{num}] descartado.", bot)
        elif acao in ("ok","envia"):
            item["status"] = "enviado"
            await send(f"✅ Abra o Gmail e envie:\n_{item.get('assunto','')}_\nhttps://mail.google.com/mail/u/0/#drafts", bot)
        with open(PENDENTES_FILE, "w") as f:
            json.dump(pendentes, f, ensure_ascii=False, indent=2)

    else:
        await send("🤔 Não entendi. Exemplos:\n• _reunião com Tiago sexta às 14h_\n• _ligar para o contador amanhã_\n• _envia email para joao@empresa.com assunto: Proposta_\n• _atas es_", bot)

# ── Main ────────────────────────────────────────────────────────────────────────
async def run_bot():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    logger.info("Bot RAN iniciado via polling")
    async with app:
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await asyncio.Event().wait()

if __name__ == "__main__":
    start_health_server()
    asyncio.run(run_bot())
