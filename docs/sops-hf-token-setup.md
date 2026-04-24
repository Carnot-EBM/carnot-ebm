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
  - path_regex: secrets/.*\.yaml$
    age: age1abc...xyz
```

Commit .sops.yaml to git (it contains only the public key — safe to share):

```bash
git add .sops.yaml
git commit -m "Add SOPS age key config for secrets/"
```

## Step 4: Create the plaintext secrets file

```bash
mkdir -p secrets
cat > secrets/hf_token.yaml << 'EOF'
HF_TOKEN: hf_YOUR_TOKEN_HERE
EOF
```

Replace `hf_YOUR_TOKEN_HERE` with your actual HuggingFace write token from
https://huggingface.co/settings/tokens

DO NOT commit this plaintext file.

## Step 5: Encrypt the secrets file

```bash
sops -e -i secrets/hf_token.yaml
```

The file is now encrypted in-place.  Verify:

```bash
cat secrets/hf_token.yaml
# Should show SOPS envelope, not plaintext
```

Add the encrypted file to git:

```bash
git add secrets/hf_token.yaml
git commit -m "Add SOPS-encrypted HF_TOKEN"
```

secrets/hf_token.yaml is safe to commit because only holders of the age private key
(~/.config/sops/age/keys.txt) can decrypt it.

## Step 6: Inject the token into conductor sessions

Before running the research conductor or any experiment that needs HF_TOKEN:

```bash
eval $(sops -d secrets/hf_token.yaml | grep HF_TOKEN)
export HF_TOKEN
```

Or add to your shell session:

```bash
export SOPS_AGE_KEY_FILE=~/.config/sops/age/keys.txt
eval $(sops -d secrets/hf_token.yaml | grep HF_TOKEN)
```

Verify the token is available:

```bash
echo $HF_TOKEN | head -c 20
```

## Rotating the token

When your HuggingFace token expires or is revoked:

1. Decrypt, update, re-encrypt:
   ```bash
   sops secrets/hf_token.yaml
   # Edit the file in your $EDITOR, save and exit
   # SOPS re-encrypts automatically on save
   ```

2. Commit the updated encrypted file.

## Security notes

- NEVER commit ~/.config/sops/age/keys.txt or any plaintext secret
- The .gitignore should exclude secrets/*.yaml.plaintext and any .env files
- If you accidentally commit a plaintext secret, rotate the token immediately and
  use `git filter-branch` or BFG to purge the history
