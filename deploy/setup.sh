#!/usr/bin/env bash
# Set up or update LUNAVI on an Ubuntu server (Oracle Cloud Always Free works).
# Safe to run again: each step checks what is already there.
#
# Usage, from the lunavi-back checkout on the server:
#   bash deploy/setup.sh <host>     the first time
#   bash deploy/setup.sh            later runs reuse the saved host
# where <host> is a domain pointing at the server, or <ip-with-dashes>.sslip.io.
#
# The web app is built from a lunavi-front checkout next to lunavi-back when one exists,
# otherwise it is copied from an uploaded build in ~/lunavi-web.
set -euo pipefail

HOST_FILE="$HOME/.lunavi-host"
if [ -n "${1:-}" ]; then
	HOST="$1"
	echo "$HOST" > "$HOST_FILE"
elif [ -f "$HOST_FILE" ]; then
	HOST="$(cat "$HOST_FILE")"
else
	echo "usage: bash deploy/setup.sh <host>, for example 129-146-10-20.sslip.io" >&2
	exit 1
fi
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FRONT_DIR="${LUNAVI_FRONT_DIR:-$(dirname "$REPO_DIR")/lunavi-front}"
WEB_UPLOAD="$HOME/lunavi-web"
WEB_ROOT="/var/www/lunavi"
SERVICE="lunavi-api"
NODE_MAJOR=24  # Vite 8 needs Node 20.19 or newer; Ubuntu's own package is older

step() { printf '\n== %s\n' "$1"; }

step "System packages and Caddy"
if ! command -v caddy >/dev/null 2>&1; then
	sudo apt-get update
	sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gpg
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
		| sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
		| sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
	sudo chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg
	sudo chmod o+r /etc/apt/sources.list.d/caddy-stable.list
	sudo apt-get update
	sudo apt-get install -y caddy
else
	echo "Caddy is already installed."
fi

step "uv and the Python environment"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
cd "$REPO_DIR"
"$UV" sync --frozen --no-dev --python 3.12

step "Earth table"
if [ -f data/results/earth.parquet ] && [ -f data/results/manifest.json ]; then
	echo "Found data/results/earth.parquet."
else
	echo "data/results/earth.parquet is missing, so the API will serve the dummy table."
	echo "Copy earth.parquet and manifest.json into $REPO_DIR/data/results, or build them with:"
	echo "  $UV run python -m pipeline.run"
fi

step "Firewall: allow web traffic"
# Oracle's Ubuntu images block incoming ports in iptables, on top of the cloud security list.
for port in 80 443; do
	if ! sudo iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null; then
		sudo iptables -I INPUT 5 -p tcp -m state --state NEW --dport "$port" -j ACCEPT
		echo "Opened port $port."
	fi
done
if command -v netfilter-persistent >/dev/null 2>&1; then
	sudo netfilter-persistent save
else
	echo "netfilter-persistent not found: the iptables rules last until the next reboot."
fi

step "API service"
sudo tee "/etc/systemd/system/$SERVICE.service" >/dev/null <<EOF
[Unit]
Description=LUNAVI API
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null
sudo systemctl restart "$SERVICE"

step "Web app"
WEB_SOURCE=""
if [ -f "$FRONT_DIR/package.json" ]; then
	node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
	if [ "$node_major" -lt 20 ]; then
		echo "Installing Node.js $NODE_MAJOR from NodeSource."
		curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" -o /tmp/nodesource_setup.sh
		sudo -E bash /tmp/nodesource_setup.sh
		sudo apt-get install -y nodejs
	fi
	echo "Building the web app in $FRONT_DIR."
	(cd "$FRONT_DIR" && npm ci --no-audit --no-fund && npm run build)
	WEB_SOURCE="$FRONT_DIR/dist"
elif [ -f "$WEB_UPLOAD/index.html" ]; then
	WEB_SOURCE="$WEB_UPLOAD"
fi
sudo mkdir -p "$WEB_ROOT"
if [ -n "$WEB_SOURCE" ]; then
	sudo rm -rf "${WEB_ROOT:?}"/*
	sudo cp -r "$WEB_SOURCE"/. "$WEB_ROOT"/
	sudo chmod -R a+rX "$WEB_ROOT"
	echo "Installed the web app from $WEB_SOURCE."
else
	echo "No web app found. Clone lunavi-front next to lunavi-back, or upload a dist folder to $WEB_UPLOAD."
fi

step "Caddy configuration"
sed "s/__HOST__/$HOST/" "$REPO_DIR/deploy/Caddyfile" | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy || sudo systemctl restart caddy

step "Checks"
for _ in $(seq 1 20); do
	if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
	sleep 1
done
echo "API: $(curl -fsS http://127.0.0.1:8000/health || echo 'not answering, see: sudo journalctl -u lunavi-api -n 50')"
echo
echo "Open https://$HOST in a browser. The first visit can take a minute while Caddy gets the certificate."
