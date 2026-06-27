# Running Claude Code Autonomously in a Locked-Down Container

This guide runs Claude Code with `--dangerously-skip-permissions` (full autonomy:
it can edit, create, and delete any file and run any command **inside the
container**) while making sure it **cannot** touch your host, your Docker daemon,
your SSH/cloud credentials, or the open internet.

You get two containers wired together with Docker Compose:

- **`claude`** — runs Claude Code as a non-root user. Has **no direct internet**.
- **`proxy`** — a Squid egress proxy that is the **only** path to the internet, and
  only to a fixed allow-list (Anthropic + GitHub).

**Authentication:** by default this uses your **Claude Pro/Max subscription** — you
log in once with your Claude account and no API key is needed. (An API key is an
optional alternative; see Section 4.)

All the files referenced below live in **`docker/sandbox/`**:

```
docker/sandbox/
├── docker-compose.yml      # the two services + networks + volumes
├── Dockerfile              # claude container (CUDA + Python + torch + Node + Claude Code, non-root)
├── Dockerfile.proxy        # squid container
├── squid.conf              # allow-list policy
├── allowed_domains.txt     # the 11 permitted hostnames
├── copy-in.sh / copy-in.cmd# copy a project into /workspace with correct ownership
├── .env.example            # optional: only for API-key billing instead of a subscription
└── .gitignore              # keeps .env and review-output out of git
```

You need: a machine (laptop or VPS) with **Docker** + the **`docker compose`**
plugin, and a **Claude Pro or Max subscription**.

**GPU training:** the `claude` image ships CUDA 12.1 + PyTorch (cu121,
Turing/GTX-1650 compatible) and the compose file passes the GPU into the
container. To use it you need an **NVIDIA GPU** plus the **NVIDIA Container
Toolkit** on the host (on Windows: Docker Desktop with the **WSL2** backend and a
current NVIDIA driver). Verify the GPU reaches Docker **before** the (multi-GB)
build — see Section 0. If you have no NVIDIA GPU, the box still works for code
editing; training just won't have CUDA.

---

## 0. Confirm the GPU reaches Docker (do this first)

The image build is several GB, so verify GPU passthrough works before building:

```bash
# Host has an NVIDIA GPU + driver?
nvidia-smi                 # should list your GPU (e.g. "NVIDIA GeForce GTX 1650")

# Docker can pass that GPU into a container?
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
#   should print the SAME GPU table from inside a container
```

If the second command fails: install the **NVIDIA Container Toolkit** (Linux) or
enable GPU support in **Docker Desktop + WSL2** (Windows), then retry. Don't build
until it lists your GPU.

---

## 1. One-time setup

```bash
cd docker/sandbox

# Build both images
docker compose build

# Start both containers (proxy first, then claude once proxy is healthy)
docker compose up -d
docker compose ps     # both should be "running"; proxy should be "healthy"
```

No `.env` is needed for subscription login — you'll authenticate interactively in
Section 4. If you ever change `allowed_domains.txt` or `squid.conf`, rebuild and
restart the proxy: `docker compose up -d --build proxy`.

---

## 2. Verify the network jail (do this before trusting it)

Run each check. The **expected result is in the comment** — if any check behaves
differently, stop and fix it before letting Claude run.

```bash
# Shorthand: run a command inside the claude container as the non-root user
SH='docker compose exec -T claude'

# (a) Anthropic API reachable THROUGH the proxy  -> expect HTTP 401 (auth works,
#     key not sent by curl) which proves the TLS connection succeeded.
$SH curl -s -o /dev/null -w '%{http_code}\n' https://api.anthropic.com/v1/models
#     expect: 401   (NOT 000 / not a timeout)

# (b) claude.ai reachable through the proxy (needed for subscription login).
$SH curl -s -o /dev/null -w '%{http_code}\n' https://claude.ai
#     expect: 200 (or a 3xx redirect) — NOT 000 / not 403

# (c) GitHub reachable through the proxy -> expect 200
$SH curl -s -o /dev/null -w '%{http_code}\n' https://api.github.com
#     expect: 200

# (d) A random site is BLOCKED by the proxy -> Squid returns 403
$SH curl -s -o /dev/null -w '%{http_code}\n' https://example.com
#     expect: 403   (denied by Squid allow-list)

# (e) Direct internet, BYPASSING the proxy, FAILS -> no route, hangs/refused.
#     --noproxy '*' ignores the HTTP(S)_PROXY env vars.
$SH curl -s --noproxy '*' --max-time 8 -o /dev/null -w '%{http_code}\n' https://example.com
#     expect: 000  (connect timeout / no route — the internal network has no NAT)

# (f) Direct hit to an ALLOWED host without the proxy also fails (proves the
#     allow-list isn't a loophole — there's simply no direct egress at all).
$SH curl -s --noproxy '*' --max-time 8 -o /dev/null -w '%{http_code}\n' https://api.anthropic.com
#     expect: 000
```

Watch what the proxy is allowing/denying in real time:

```bash
docker compose logs -f proxy     # TCP_DENIED lines = blocked attempts
```

---

## 3. Copy your project IN (no host bind mount)

The workspace is a Docker **named volume**, not a mount of a host directory, so
Claude can never walk up out of `/workspace` into your real filesystem.

**Use the helper script** (from `docker/sandbox/`) — it copies the project in
**and** fixes file ownership in one step:

```bash
# Linux / macOS
./copy-in.sh /path/to/my-project

# Windows
copy-in C:\path\to\my-project
```

Why a helper instead of plain `docker compose cp`? `cp` writes files into the
volume as **root**, but the sandbox runs as the non-root `claude` user — and
because the container has `cap_drop: ALL` (no `CAP_CHOWN`), you can't fix the
ownership *inside* the running container. The helper does the `cp`, then chowns
the files using a **throwaway root container** (default capabilities), so the
running sandbox stays fully locked (`cap_drop: ALL`, non-root). Run it once per
project you copy in.

<details><summary>Manual equivalent (what the helper runs)</summary>

```bash
docker compose cp ./my-project claude:/workspace/my-project
docker run --rm -u 0:0 -v sandbox_workspace:/workspace sandbox-claude \
    chown -R claude:claude /workspace/my-project
```
(`sandbox_workspace` / `sandbox-claude` are the default volume/image names when
the compose project directory is `sandbox`.)
</details>

**If the project is a Python package (editable install), do it offline once.**
All dependencies are baked into the image, and the network allow-list blocks
PyPI, so install with `--no-deps --no-build-isolation` (no PyPI fetch) as the
non-root user:

```bash
docker compose exec claude bash -lc \
  'cd /workspace/my-project && pip install -e . --no-deps --no-build-isolation --user'
```

Confirm Python, torch, and the GPU are all visible inside the container:

```bash
docker compose exec claude python -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
#   expect: torch 2.4.1+cu121 cuda True  /  NVIDIA GeForce GTX 1650
```

If `cuda True` and your GPU name print, training is ready.

---

## 4. Log in (one-time, interactive)

The container has no browser, so you do a one-time **manual** OAuth login. Your
credentials and folder-trust live under `/home/claude` (both `~/.claude/` and
`~/.claude.json`), which is the `claude_config` named volume — so login and trust
survive restarts and rebuilds; you only do this once.

```bash
# Open a shell in the sandbox as the non-root user
docker compose exec claude bash

# ...now inside the container, start Claude and trigger login:
claude
#   - choose "Log in with your Claude account" (the Pro/Max option,
#     NOT the API-key option)
#   - Claude prints a URL. Copy it, open it in YOUR host browser, approve,
#     then paste the resulting code back into the container prompt.
```

This works because `claude.ai` and `api.anthropic.com` are on the proxy
allow-list. The login flow never needs the container to open a browser itself.

> **Optional — use an API key instead of a subscription.** If you'd rather pay
> per-token: `cp .env.example .env`, put your key in it, set
> `ANTHROPIC_API_KEY=sk-ant-...`, then `docker compose up -d` to recreate the
> container with the key. Skip the login step above. (Note the API-key warnings
> in the last section.)

---

## 5. Run Claude Code autonomously

```bash
# Inside the sandbox shell (docker compose exec claude bash):
cd /workspace/my-project
claude --dangerously-skip-permissions
```

That's the exact command. Inside, Claude can read/write/delete anything under
`/workspace` and run any shell command, but only reach the allow-listed hosts.

For a fully unattended run, give it the task non-interactively:

```bash
docker compose exec claude bash -lc \
  'cd /workspace/my-project && claude --dangerously-skip-permissions -p "Run the autoresearch loop per CLAUDE.md"'
```

---

## 6. Copy the finished work OUT for review

Never copy blindly back over your real repo. Pull the result into a throwaway
directory and diff it first.

```bash
# Pull the worked copy out of the volume into a review folder on the host
docker compose -f docker/sandbox/docker-compose.yml cp \
    claude:/workspace/my-project ./review-output

# Compare against your real repo before accepting anything
diff -ruN ./my-project ./review-output | less
#   or, if your real repo is a git checkout:
git -C ./my-project --no-index diff ./my-project ./review-output
```

Only after you've read the diff should you copy approved changes back into your
real working tree.

---

## 7. Teardown

```bash
docker compose down                 # stop containers, KEEP both named volumes
docker compose down -v              # also DELETE the volumes: wipes work AND your login
```

`down` (without `-v`) keeps both the `workspace` and `claude_config` volumes, so
your work and your subscription login both persist. `down -v` deletes them — you'd
re-copy the project and log in again.

---

## What this setup blocks

- **No host filesystem access.** `/workspace` is a named volume; there is no bind
  mount of any host directory. Claude cannot read your home dir, SSH keys, dotfiles,
  or `/etc`.
- **No Docker daemon access.** `/var/run/docker.sock` is **not** mounted and the
  Docker CLI is not installed, so Claude can't start sibling containers or escape.
- **GPU access does not weaken this.** The NVIDIA runtime exposes only the GPU
  device + driver libraries to the container; it grants no host filesystem, root,
  or docker-socket access. All the other limits above still apply.
- **No PyPI / HuggingFace / general downloads.** Only the 11 hosts below are
  reachable, so `pip install` from PyPI and dataset/model downloads from
  HuggingFace will fail. All Python deps are pre-baked into the image; any
  training data must be copied into `/workspace` beforehand.
- **No host network.** The `claude` container sits on an `internal: true` Docker
  network with no NAT to the internet. It can only talk to the proxy.
- **No root / no privilege escalation.** Runs as the non-root `claude` user,
  `cap_drop: ALL` removes all Linux capabilities, and `no-new-privileges:true`
  blocks setuid escalation. There is no `sudo` in the image.
- **No open internet.** The proxy permits **only** these 11 hosts; every other
  destination gets a `403` from Squid:
  `api.anthropic.com`, `claude.ai`, `platform.claude.com`, `github.com`,
  `api.github.com`, `raw.githubusercontent.com`, `objects.githubusercontent.com`,
  `codeload.github.com`, `lfs.github.com`, `github-cloud.githubusercontent.com`,
  `github-cloud.s3.amazonaws.com`. (`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`
  keeps telemetry/auto-update off so nothing else is needed.)
- **No resource exhaustion of the host.** `mem_limit: 4g` (+ `memswap_limit` = no
  swap), `cpus: 2.0`, and `pids_limit: 512` cap RAM, CPU, and process/fork count.
- **No SSH/cloud creds.** None are mounted or present in the image. The only
  credential inside is your Claude login (subscription token, or an API key if you
  chose that option).

## What this setup does NOT block — read this

This sandbox contains Claude; it does not make Claude harmless.

- ⚠️ **Claude can read your login credentials.** Your subscription token lives in
  `/home/claude` (or the API key in the environment, if you chose that).
  Anything running inside the container can read it. It's bounded to *this* account.
- ⚠️ **Claude can modify or delete anything in `/workspace`.** That's the point of
  `--dangerously-skip-permissions`. Anything you copy in is fair game, including
  destructive changes. The named volume is your only copy until you copy results out.
- ⚠️ **Claude can send your code and file contents to Anthropic.** Everything in the
  workspace it reads can be transmitted to `api.anthropic.com`. Don't put secrets,
  customer data, or anything you wouldn't send to the API into `/workspace`.
- ⚠️ **Claude can spend your quota / money.** Autonomous loops make many calls —
  they burn your Pro/Max usage limits (or, in API-key mode, real money; set a spend
  limit and watch usage).
- ⚠️ **GitHub is reachable.** If you provide a GitHub token in the workspace, Claude
  can push to and read from repos that token allows. Scope tokens narrowly.

**Always review the diff (Section 6) before copying changes back to your real
repository.** The container limits blast radius to the volume; your review is what
protects the real repo.

---

## Quick reference

| Action | Command |
|---|---|
| Build & start | `docker compose up -d --build` |
| Status | `docker compose ps` |
| Watch egress decisions | `docker compose logs -f proxy` |
| Shell into sandbox | `docker compose exec claude bash` |
| Log in (one-time) | `claude` → "Log in with your Claude account" |
| Run Claude autonomously | `claude --dangerously-skip-permissions` |
| Copy project in (+fix perms) | `./copy-in.sh ./proj`  (Win: `copy-in .\proj`) |
| Copy results out | `docker compose cp claude:/workspace/proj ./review-output` |
| GPU check inside | `docker compose exec claude python -c "import torch;print(torch.cuda.is_available())"` |
| Stop (keep work + login) | `docker compose down` |
| Stop & wipe volumes | `docker compose down -v` |

---

## Troubleshooting (issues you may hit, and the fixes)

- **`failed to read dockerfile ... no such file or directory` / `Dockerfile`
  transfers as `2B`.** You're building from a broken/partial copy of the repo
  (e.g. a browser ZIP that saved an empty `Dockerfile`). Get the files via `git
  clone` (not "Download ZIP"/"Save As"), and on Windows make sure the file is
  literally `Dockerfile` with no hidden `.txt` extension.

- **Proxy container is `unhealthy`; logs show
  `FATAL: Don't run Squid as root` or `Cannot open '/dev/stdout'`.** Fixed in
  this repo: squid runs as the non-root `proxy` user and logs to files that the
  entrypoint tails to stdout. If you edited `squid.conf`, don't set
  `cache_effective_user root` — Squid 5 refuses to run as root.

- **Claude Code: `Unable to connect to Anthropic services` /
  `Failed to connect to platform.claude.com: ERR_SOCKET_CLOSED`.** Claude Code
  pings `platform.claude.com` at startup; it must be in `allowed_domains.txt`
  (it is, in this repo). If a *different* Anthropic/GitHub host is blocked, add
  it to `allowed_domains.txt` and `docker compose up -d --build proxy`.

- **`torch.cuda.is_available()` is `False`.** The host can't pass the GPU to
  Docker. Re-run Section 0's `docker run --gpus all ... nvidia-smi`; install the
  NVIDIA Container Toolkit (Linux) or enable GPU in Docker Desktop + WSL2
  (Windows), then `docker compose up -d --build`.

- **Claude can't write files / "permission denied" in `/workspace`.** The copied
  files are owned by root. Use `./copy-in.sh` (Section 3) which fixes ownership.
  To repair an existing copy:
  `docker run --rm -u 0:0 -v sandbox_workspace:/workspace sandbox-claude chown -R claude:claude /workspace`

- **"Do you trust the files in this folder?" prompt.** This is Claude Code's
  one-time folder-trust gate (separate from `--dangerously-skip-permissions`).
  Select **Yes**; the choice is saved in the `claude_config` volume.

- **`pip install` / HuggingFace download fails.** Expected — only Anthropic +
  GitHub are reachable. Python deps are pre-baked into the image; copy any
  training data into `/workspace` instead of downloading it.
