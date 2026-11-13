# Deploy LUNAVI on Oracle Cloud (Always Free)

One small server runs everything. Caddy serves the web app, forwards `/api` to the FastAPI backend, and gets an HTTPS certificate on its own. No domain is needed: an `sslip.io` name built from the server's IP works.

```
browser ── https://<host> ── Caddy ─┬─ /        web app files (/var/www/lunavi)
                                    └─ /api/*   FastAPI on 127.0.0.1:8000
```

Both repositories are cloned on the server, side by side. The setup script builds the web app there, so a `git pull` and one command deploy any change.

## 1. Create the server (Oracle Cloud console)

1. **Compute > Instances > Create instance.**
   - Image: **Canonical Ubuntu 24.04**.
   - Shape: **VM.Standard.A1.Flex** (Ampere, Always Free eligible), 1 OCPU and 6 GB of memory is plenty. If Oracle reports no capacity, try another availability domain, or use **VM.Standard.E2.1.Micro** (1 GB of memory: slower, and building the web app may need a swap file, see Troubleshooting).
   - Networking: keep **Assign a public IPv4 address** on.
   - SSH keys: download the private key, or paste your own public key.
2. **Open ports 80 and 443.** On the instance page, open its subnet, then its **Security List**, and add two **Ingress Rules**: source `0.0.0.0/0`, protocol TCP, destination ports `80` and `443`. (Port 22 is open by default.)
3. Note the **public IP**, for example `129.146.10.20`. Your host name is then `129-146-10-20.sslip.io`.

## 2. Give the server read access to both repositories

The repositories are private. GitHub lets a deploy key open only one repository, so the server gets two read-only keys and a short SSH config that picks the right one.

```bash
ssh -i <private-key> ubuntu@<public-ip>

ssh-keygen -t ed25519 -N "" -C "lunavi-back deploy" -f ~/.ssh/lunavi_back
ssh-keygen -t ed25519 -N "" -C "lunavi-front deploy" -f ~/.ssh/lunavi_front
cat >> ~/.ssh/config <<'EOF'
Host github-lunavi-back
  HostName github.com
  User git
  IdentityFile ~/.ssh/lunavi_back
  IdentitiesOnly yes
Host github-lunavi-front
  HostName github.com
  User git
  IdentityFile ~/.ssh/lunavi_front
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
cat ~/.ssh/lunavi_back.pub    # add to lunavi-back
cat ~/.ssh/lunavi_front.pub   # add to lunavi-front
```

On GitHub, in each repository open **Settings > Deploy keys > Add deploy key**, paste the matching key, and leave write access off. Then clone both, next to each other:

```bash
git clone git@github-lunavi-back:fm1299/lunavi-back.git ~/lunavi-back
git clone git@github-lunavi-front:fm1299/lunavi-front.git ~/lunavi-front
```

## 3. Upload the Earth table (from your laptop)

The Earth table is git-ignored, so copy it from a machine that has it:

```powershell
# in lunavi-back on the laptop
scp -i <private-key> data/results/earth.parquet data/results/manifest.json ubuntu@<public-ip>:~/
```

Then on the server:

```bash
mkdir -p ~/lunavi-back/data/results
mv ~/earth.parquet ~/manifest.json ~/lunavi-back/data/results/
```

(Or build it on the server with `uv run python -m pipeline.run` after step 4, about 20 minutes and about 3 GB of downloads.)

## 4. Run the setup

```bash
cd ~/lunavi-back
bash deploy/setup.sh 129-146-10-20.sslip.io
```

It installs Caddy, uv and Node.js 24, sets up Python 3.12 and the dependencies, opens ports 80 and 443 in the server's firewall, starts the API as a service, builds and installs the web app, and configures Caddy. It saves the host name for later runs. Then open `https://129-146-10-20.sslip.io`.

## Updating

After pushing changes to GitHub, on the server:

```bash
git -C ~/lunavi-back pull
git -C ~/lunavi-front pull
bash ~/lunavi-back/deploy/setup.sh
```

That restarts the API and rebuilds the web app. To replace the Earth table, upload new `earth.parquet` and `manifest.json` into `~/lunavi-back/data/results`, then run `sudo systemctl restart lunavi-api`.

## Troubleshooting

| Symptom | Check |
|---|---|
| The page does not load at all | Security list rules for 80 and 443 (step 1.2); `sudo iptables -L INPUT -n` on the server |
| Certificate errors on the first visit | Wait a minute; `sudo journalctl -u caddy -n 50` |
| The app says it can't reach its API | `sudo systemctl status lunavi-api` and `sudo journalctl -u lunavi-api -n 50` |
| The badge says "Dummy data" | `earth.parquet` and `manifest.json` are not in `~/lunavi-back/data/results` |
| `git clone` asks for a password or says "Repository not found" | The deploy key is missing on GitHub, or the clone URL does not use the `github-lunavi-…` host from step 2 |
| The web build is killed on a 1 GB server | Add swap: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| Picking a Moon or Mars spot fails | The server needs outbound internet to read LOLA, MOLA, Diviner and the Mars Climate Database |
