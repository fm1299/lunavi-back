# Deploy LUNAVI on Oracle Cloud (Always Free)

One small server runs everything. Caddy serves the web app, forwards `/api` to the FastAPI backend, and gets an HTTPS certificate on its own. No domain is needed: an `sslip.io` name built from the server's IP works.

```
browser ── https://<host> ── Caddy ─┬─ /        web app files (/var/www/lunavi)
                                    └─ /api/*   FastAPI on 127.0.0.1:8000
```

## 1. Create the server (Oracle Cloud console)

1. **Compute > Instances > Create instance.**
   - Image: **Canonical Ubuntu 24.04**.
   - Shape: **VM.Standard.A1.Flex** (Ampere, Always Free eligible), 1 OCPU and 6 GB of memory is plenty. If Oracle reports no capacity, try another availability domain, or use **VM.Standard.E2.1.Micro** (1 GB of memory, slower but enough for a demo).
   - Networking: keep **Assign a public IPv4 address** on.
   - SSH keys: download the private key, or paste your own public key.
2. **Open ports 80 and 443.** On the instance page, open its subnet, then its **Security List**, and add two **Ingress Rules**: source `0.0.0.0/0`, protocol TCP, destination ports `80` and `443`. (Port 22 is open by default.)
3. Note the **public IP**, for example `129.146.10.20`. Your host name is then `129-146-10-20.sslip.io`.

## 2. Give the server read access to the repository

The repository is private, so the server uses a read-only deploy key.

```bash
ssh -i <private-key> ubuntu@<public-ip>
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

On GitHub, open **lunavi-back > Settings > Deploy keys > Add deploy key**, paste the key, leave write access off. Then on the server:

```bash
git clone git@github.com:fm1299/lunavi-back.git ~/lunavi-back
```

## 3. Upload the Earth table and the web app (from your laptop)

The Earth table is git-ignored, so copy it from a machine that has it. The web app is built on the laptop, so the server needs no Node.js.

```powershell
# in lunavi-back
scp -i <private-key> data/results/earth.parquet data/results/manifest.json ubuntu@<public-ip>:~/
# in lunavi-front (production builds call the API at /api, see .env.production)
npm ci
npm run build
scp -i <private-key> -r dist ubuntu@<public-ip>:~/lunavi-web
```

On the server, put the Earth table in place:

```bash
mkdir -p ~/lunavi-back/data/results
mv ~/earth.parquet ~/manifest.json ~/lunavi-back/data/results/
```

## 4. Run the setup

```bash
cd ~/lunavi-back
bash deploy/setup.sh 129-146-10-20.sslip.io
```

It installs Caddy and uv, sets up Python 3.12 and the dependencies, opens ports 80 and 443 in the server's firewall, starts the API as a service, installs the web app and configures Caddy. Then open `https://129-146-10-20.sslip.io`.

## Updating

- **Backend code:** `cd ~/lunavi-back && git pull && bash deploy/setup.sh <host>`
- **Web app:** build again on the laptop, upload `dist` to `~/lunavi-web` (delete the old folder first), then run `bash deploy/setup.sh <host>`.
- **Earth table:** upload new `earth.parquet` and `manifest.json` into `~/lunavi-back/data/results`, then `sudo systemctl restart lunavi-api`.

## Troubleshooting

| Symptom | Check |
|---|---|
| The page does not load at all | Security list rules for 80 and 443 (step 1.2); `sudo iptables -L INPUT -n` on the server |
| Certificate errors on the first visit | Wait a minute; `sudo journalctl -u caddy -n 50` |
| The app says it can't reach its API | `sudo systemctl status lunavi-api` and `sudo journalctl -u lunavi-api -n 50` |
| The badge says "Dummy data" | `earth.parquet` and `manifest.json` are not in `~/lunavi-back/data/results` |
| Picking a Moon or Mars spot fails | The server needs outbound internet to read LOLA, MOLA, Diviner and the Mars Climate Database |
