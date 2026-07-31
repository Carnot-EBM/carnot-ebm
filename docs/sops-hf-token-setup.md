# SOPS HuggingFace Token Setup

This document explains how to store the HuggingFace authentication token (HF_TOKEN)
securely using SOPS encryption so that the research conductor can publish model
artifacts without storing plaintext secrets in the repository.

SOPS (Secrets OPerationS) encrypts secret files using age or PGP keys.  The encrypted
file can be committed to git safely — only holders of the private key can decrypt it.

## Step 1: Install SOPS and age

On Arch Linux / CachyOS:

```bash
sudo pacman -S sops age
```

On Ubuntu / Debian:

```bash
sudo apt install sops age
```

Verify both tools are installed:

```bash
sops --version
age --version
```

## Step 2: Generate an age key pair

age uses Curve25519 key pairs.  The private key stays on your machine; the public key
goes into .sops.yaml so anyone with the private key can decrypt the repo's secrets.

```bash
mkdir -p ~/.config/sops/age
age-keygen -o ~/.config/sops/age/keys.txt
```

The output looks like:

```
Public key: age1abc...xyz
```

Copy the public key — you need it in Step 3.

## Step 3: Create .sops.yaml in the project root

.sops.yaml tells SOPS which key to use when encrypting new files in this repo.
Replace `age1abc...xyz` with your actual public key from Step 2.

```yaml
creation_rules:
  - path_regex: '(secrets\..*\.yaml|.*\.enc\.yaml)$'
    age: age1abc...xyz
```

Match the repo's committed `.sops.yaml` exactly.  Note this keys off the `.enc.yaml`
suffix, not the `secrets/` directory — which is why Step 4 names the file
`hf_token.enc.yaml` from the start.

Commit .sops.yaml to git (it contains only the public key — safe to share):

```bash
git add .sops.yaml
git commit -m "Add SOPS age key config for secrets/"
```

## Step 4: Create the secrets file

Write it directly at the `.enc.yaml` name, even though it starts out as plaintext.
Two reasons, both load-bearing:

- `.sops.yaml`'s `creation_rules` match `secrets\..*\.yaml` or `.*\.enc\.yaml`.  A file
  called `secrets/hf_token.yaml` matches **neither** (the first pattern needs a literal
  `secrets.` prefix with a dot, not the `secrets/` directory), so `sops -e` would fail
  with "no matching creation rules".
- `.gitignore` denies all of `secrets/*` and re-admits only `*.enc.yaml` / `*.enc`.  A
  plaintext `secrets/hf_token.yaml` is therefore unstageable — which is the point, but
  it also means a workflow built around that name simply doesn't work.

```bash
mkdir -p secrets
cat > secrets/hf_token.enc.yaml << 'EOF'
HF_TOKEN: hf_YOUR_TOKEN_HERE
EOF
```

Replace `hf_YOUR_TOKEN_HERE` with your actual HuggingFace write token from
https://huggingface.co/settings/tokens

The file is plaintext until Step 5 encrypts it in place.  Do not commit in between.

## Step 5: Encrypt the secrets file

```bash
sops -e -i secrets/hf_token.enc.yaml
```

The file is now encrypted in-place.  Verify BEFORE staging — this is the step that
catches a failed encryption, and staging first is how plaintext escapes:

```bash
grep -q 'ENC\[AES256_GCM' secrets/hf_token.enc.yaml \
  && echo "OK: sops envelope present" \
  || echo "STOP: still plaintext, do not commit"
```

Add the encrypted file to git:

```bash
git add secrets/hf_token.enc.yaml
git commit -m "Add SOPS-encrypted HF_TOKEN"
```

secrets/hf_token.enc.yaml is safe to commit because only holders of the age private key
(~/.config/sops/age/keys.txt) can decrypt it.

## Step 6: Inject the token into conductor sessions

Before running the research conductor or any experiment that needs HF_TOKEN:

```bash
eval $(sops -d secrets/hf_token.enc.yaml | grep HF_TOKEN)
export HF_TOKEN
```

Or add to your shell session:

```bash
export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt
eval $(sops -d secrets/hf_token.enc.yaml | grep HF_TOKEN)
```

Verify the token is available:

```bash
echo $HF_TOKEN | head -c 20
```

## Rotating the token

When your HuggingFace token expires or is revoked:

1. Decrypt, update, re-encrypt:
   ```bash
   sops secrets/hf_token.enc.yaml
   # Edit the file in your $EDITOR, save and exit
   # SOPS re-encrypts automatically on save
   ```

2. Commit the updated encrypted file.

## Security notes

- NEVER commit ~/.config/sops/age/keys.txt or any plaintext secret
- `.gitignore` denies `secrets/*` and re-admits only `*.enc.yaml` / `*.enc`, so a
  decrypted file left in `secrets/` cannot be staged.  It also denies `ops/secrets.yaml`
  and extensionless SSH private keys (`id_rsa`, `id_ed25519`, ...), which `*.key` /
  `*.pem` never matched.  Verify any new secret path with
  `git check-ignore -v <path>` before writing to it.
- The `gitleaks` pre-commit hook scans staged diffs, but it only sees what reaches
  `git add` — the ignore rules above are the layer that keeps it from getting there.
- If you accidentally commit a plaintext secret, rotate the token immediately and
  use `git filter-branch` or BFG to purge the history
