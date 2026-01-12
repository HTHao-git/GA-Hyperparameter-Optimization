# Kaggle API Setup Guide

This document explains how to set up Kaggle API credentials for downloading datasets.

**Updated:** Kaggle now offers TWO authentication methods! 

---

## Authentication Methods

### Method 1: Token-Based (NEW - Recommended) ✨

**Pros:**
- ✅ Simpler setup
- ✅ No files to manage
- ✅ Easy to rotate/revoke
- ✅ Works across multiple projects

**Cons:**
- ⚠️ Need to set environment variable
- ⚠️ May need to restart terminal/IDE

---

### Method 2: Legacy kaggle.json File (OLD)

**Pros:**
- ✅ Persistent (no env vars needed)
- ✅ Works with older Kaggle API versions

**Cons:**
- ⚠️ File-based (must protect from accidental commits)
- ⚠️ Less convenient to rotate

---

## Setup Instructions

### 🆕 Method 1: Token-Based Authentication

#### Step 1: Get Your Token

1. Go to: https://www.kaggle.com/
2. Login → Click profile picture → **Settings**
3. Scroll to **API** section
4. Click **"Create New Token"**
5. **Copy the token** (looks like:  `a1b2c3d4e5f6g7h8i9j0... `)

#### Step 2: Set Environment Variable

**Windows (PowerShell) - Temporary (current session only):**

```powershell
$env:KAGGLE_API_TOKEN = "your_token_here"
```

**Windows (Command Prompt) - Temporary:**

```cmd
set KAGGLE_API_TOKEN=your_token_here
```

**Windows (Permanent) - System Environment Variables:**

1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to **Advanced** tab → **Environment Variables**
3. Under **User variables**, click **New**
4. Variable name: `KAGGLE_API_TOKEN`
5. Variable value: `your_token_here`
6. Click **OK** to save
7. **Restart your terminal/IDE**

**macOS/Linux (Temporary):**

```bash
export KAGGLE_API_TOKEN="your_token_here"
```

**macOS/Linux (Permanent) - Add to shell profile:**

```bash
# For bash (add to ~/. bashrc or ~/.bash_profile):
echo 'export KAGGLE_API_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc

# For zsh (add to ~/. zshrc):
echo 'export KAGGLE_API_TOKEN="your_token_here"' >> ~/.zshrc
source ~/. zshrc
```

#### Step 3: Verify

**In Python:**

```python
import os
print(os.environ.get('KAGGLE_API_TOKEN'))
# Should print your token
```

**Or test the loader:**

```cmd
python test_kaggle_setup.py
```

---

### 📜 Method 2: Legacy kaggle.json File

#### Step 1: Get kaggle.json

1. Go to: https://www.kaggle.com/
2. Login → Settings → **API**
3. Look for **"Legacy API Tokens"** section
4. Click **"Create New Token"** (under legacy)
5. This downloads `kaggle.json` to your Downloads folder

**File contents:**
```json
{
  "username": "your_kaggle_username",
  "key": "abc123..."
}
```

#### Step 2: Install kaggle.json

**Windows:**

```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

**macOS/Linux:**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/. kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Verify

```cmd
kaggle datasets list
```

If you see a list of datasets, you're set!  ✅

---

## Security Notes

### For Token Method: 

- ✅ Never commit tokens to Git
- ✅ Use environment variables (not hardcoded in code)
- ✅ Rotate tokens periodically
- ✅ Revoke tokens if exposed

### For Legacy Method:

- ✅ `.gitignore` excludes `kaggle.json` and `.kaggle/` directory
- ✅ Never commit `kaggle.json` to Git
- ✅ Treat it like a password

---

## Troubleshooting

### "Kaggle API credentials not found"

**Token method:**
- Check:  `echo %KAGGLE_API_TOKEN%` (Windows) or `echo $KAGGLE_API_TOKEN` (Unix)
- If empty, token not set
- Try restarting terminal/IDE after setting

**Legacy method:**
- Check file exists: `dir %USERPROFILE%\.kaggle\kaggle.json` (Windows)
- Check file exists: `ls ~/.kaggle/kaggle.json` (Unix)

### "401 Unauthorized"

- Token/key expired or invalid
- Create a new token/file

### "403 Forbidden"

- Accept dataset terms on Kaggle website first
- Visit dataset page → Click "Download"

---

## Which Method Should I Use?

**Recommendation:  Token-based (Method 1)**

- Easier to manage
- More secure (easier to rotate)
- Modern standard

**Use legacy (Method 2) if:**
- You prefer file-based auth
- You have existing kaggle.json file
- You're using older Kaggle API

---

## Testing Your Setup

Run this test script:

```cmd
python test_kaggle_setup.py
```

**Expected output:**

```
======================================================================
KAGGLE API TEST
======================================================================
[SUCCESS] Kaggle API authenticated (token method)
✓ Kaggle API is configured and working! 

You can now download datasets from Kaggle
Example: Steel Plates Faults dataset
```

---

## Disabling Kaggle (Optional)

If you don't want to use Kaggle:

- Don't set up credentials
- Data loader will skip Kaggle sources automatically
- Use alternative sources (UCI, GitHub) instead

---

## Privacy Checklist

- ✅ `.gitignore` excludes sensitive files
- ✅ Never share tokens/keys
- ✅ Use environment variables (not hardcoded)
- ✅ Rotate credentials if exposed