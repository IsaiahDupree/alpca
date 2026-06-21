"""
Send the Alpca package email (with attachments) from isaiahdupree33@gmail.com
via the Gmail API, using the OAuth refresh token in agent-comms/.env.

Does NOT send unless run with --send. Default is a dry-run that prints the plan.

Run:
  python scripts/send_package_email.py            # dry-run (prints plan)
  python scripts/send_package_email.py --send     # actually send
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys
from email.message import EmailMessage
from pathlib import Path

import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
AGENT_COMMS_ENV = Path("/Users/isaiahdupree/Documents/Software/agent-comms/.env")

TO = "brett.finnell@gmail.com"
CC = "isaiahdupree33@gmail.com"
SUBJECT = "Alpca — honest strategy-evaluation platform (code + results + AI-IDE setup)"
BODY = (ROOT / "docs" / "_email_body.txt").read_text()
ATTACHMENTS = [
    ROOT / "docs" / "STATE_OF_THE_PROGRAM.docx",
    ROOT / "docs" / "GETTING_STARTED_WITH_AI.docx",
    ROOT / "docs" / "EDGE_CASE_STUDIES.docx",
    ROOT / "docs" / "strategy_landscape.png",
    ROOT / "docs" / "deployed_results.png",
]


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def access_token(env: dict) -> str:
    data = urllib.parse.urlencode({
        "client_id": env["GOOGLE_CLIENT_ID"],
        "client_secret": env["GOOGLE_CLIENT_SECRET"],
        "refresh_token": env["GOOGLE_REFRESH_TOKEN"],
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
    with urllib.request.urlopen(req, timeout=30) as r:
        import json
        return json.load(r)["access_token"]


def build_message(sender: str) -> str:
    msg = EmailMessage()
    msg["To"] = TO
    if CC:
        msg["Cc"] = CC
    msg["From"] = sender
    msg["Subject"] = SUBJECT
    msg.set_content(BODY)
    for p in ATTACHMENTS:
        if not p.exists():
            raise FileNotFoundError(p)
        ctype, _ = mimetypes.guess_type(str(p))
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype,
                           filename=p.name)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def main():
    send = "--send" in sys.argv
    env = load_env(AGENT_COMMS_ENV)
    sender = env.get("AGENT_EMAIL", "isaiahdupree33@gmail.com")
    print(f"From:    {sender}")
    print(f"To:      {TO}")
    print(f"Cc:      {CC}")
    print(f"Subject: {SUBJECT}")
    print(f"Attach:  {[p.name for p in ATTACHMENTS]}")
    total = sum(p.stat().st_size for p in ATTACHMENTS if p.exists())
    print(f"Size:    {total/1024:.0f} KB")
    if not send:
        print("\n[DRY-RUN] Not sent. Re-run with --send to deliver.")
        return
    raw = build_message(sender)
    token = access_token(env)
    import json
    payload = json.dumps({"raw": raw}).encode()
    req = urllib.request.Request(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        data=payload,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    print(f"\n[SENT] message id={out.get('id')} threadId={out.get('threadId')}")


if __name__ == "__main__":
    main()
