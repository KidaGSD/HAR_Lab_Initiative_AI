# Label Studio — Collaborate on HAR Action Labeling

This guide is for collaborators joining the HAR action-labeling project. Use the invitation link below to create an account and start labeling.

---

## 1. Join the Project

**Invitation link:**  
[http://localhost:8080/user/signup/?token=qdLXG6rPWAszso4iho41vS6lLFn5vQa4E3qfE8ml](http://localhost:8080/user/signup/?token=qdLXG6rPWAszso4iho41vS6lLFn5vQa4E3qfE8ml)

1. Open the link in your browser (Chrome recommended).
2. Create an account with your email and password.
3. Log in and open the **HAR_dataset** project from the project list.

> If the link does not work (e.g. you are not on the same network), ask the project owner for the correct Label Studio URL.

---

## 2. How to Label

Each task shows a narration (activity description) and metadata. Choose one label per row:

| Label | When to use |
|-------|-------------|
| **Gold** | The action label is correct and high quality. |
| **Bad** | The action label is wrong or low quality. |
| **Skip** | Skip this task (e.g. unclear, edge case). |
| **Delete Row** | The row should be removed from the dataset. |

**Keyboard shortcuts:** `1` Gold, `2` Bad, `3` Skip, `4` Delete Row.

---

## 3. Check Progress (Optional)

To see the current gold/bad ratio from annotations:

```bash
export LABEL_STUDIO_API_KEY="<your-token>"
python scripts/labelstudio_gold_bad_ratio.py
```

Get your API token: **Account & Settings** (top right) → **Legacy Tokens** → create and copy.

---

## 4. Install Label Studio (If You Need to Run It)

Only needed if you run your own Label Studio instance (e.g. for local development).

```bash
pip install label-studio
label-studio
```

Opens at http://localhost:8080.
