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
├── Dockerfile              # claude container (Node + Claude Code, non-root)
├── Dockerfile.proxy        # squid container
├── squid.conf              # allow-list policy
├── allowed_domains.txt     # the 10 permitted hostnames
├── .env.example            # optional: only for API-key billing instead of a subscription
└── .gitignore              # keeps .env and review-output out of git
```

You need: a Linux/macOS/WSL2 machine (laptop or VPS) with **Docker** and the
**`docker compose`** plugin, and a **Claude Pro or Max subscription**. No GPU is
required for this guide.

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
Claude can never walk up out of `/workspace` into your real filesystem. Use
`docker compose cp` to push a copy of your project into the volume:

```bash
# From your project's parent directory. This COPIES (does not mount) the tree.
docker compose -f docker/sandbox/docker-compose.yml cp \
    ./my-project claude:/workspace/my-project

# Fix ownership so the non-root 'claude' user can write to it
docker compose -f docker/sandbox/docker-compose.yml exec -u root claude \
    chown -R claude:claude /workspace/my-project
```

(Adjust `-f` path depending on where you run the command; if you're already in
`docker/sandbox/` you can drop the `-f ...` flag.)

---

## 4. Log in (one-time, interactive)

The container has no browser, so you do a one-time **manual** OAuth login. Your
credentials are saved to `/home/claude/.claude`, which is a named volume
(`claude_config`), so this survives restarts and rebuilds — you only do it once.

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
- **No host network.** The `claude` container sits on an `internal: true` Docker
  network with no NAT to the internet. It can only talk to the proxy.
- **No root / no privilege escalation.** Runs as the non-root `claude` user,
  `cap_drop: ALL` removes all Linux capabilities, and `no-new-privileges:true`
  blocks setuid escalation. There is no `sudo` in the image.
- **No open internet.** The proxy permits **only** these 10 hosts; every other
  destination gets a `403` from Squid:
  `api.anthropic.com`, `claude.ai`, `github.com`, `api.github.com`,
  `raw.githubusercontent.com`, `objects.githubusercontent.com`,
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
  `/home/claude/.claude` (or the API key in the environment, if you chose that).
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
| Copy project in | `docker compose cp ./proj claude:/workspace/proj` |
| Copy results out | `docker compose cp claude:/workspace/proj ./review-output` |
| Stop (keep work + login) | `docker compose down` |
| Stop & wipe volumes | `docker compose down -v` |
