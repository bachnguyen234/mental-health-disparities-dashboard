import requests
# give your server a second to start, or start this after it's already running
r = requests.get("http://127.0.0.1:8050")
with open("dashboard_snapshot.html", "w", encoding="utf-8") as f:
    f.write(r.text)