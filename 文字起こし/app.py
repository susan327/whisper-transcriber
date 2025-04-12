import os
import uuid
import whisper
from flask import Flask, render_template, request, redirect, send_file, url_for, make_response
from datetime import datetime
from docx import Document
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
HISTORY_FOLDER = "history"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

def get_user_id():
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    return user_id

def get_user_paths(user_id):
    user_folder = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(user_folder, exist_ok=True)
    history_file = os.path.join(HISTORY_FOLDER, f"history_{user_id}.json")
    return user_folder, history_file

def load_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def write_history(history, history_file):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@app.route("/", methods=["GET"])
def index():
    user_id = get_user_id()
    user_folder, history_file = get_user_paths(user_id)
    history = load_history(history_file)
    response = make_response(render_template("index.html", history=history))
    response.set_cookie("user_id", user_id)
    return response

@app.route("/transcribe", methods=["POST"])
def transcribe():
    user_id = get_user_id()
    user_folder, history_file = get_user_paths(user_id)
    audio = request.files["audio"]
    model_name = request.form.get("model", "small")
    filename = str(uuid.uuid4()) + ".docx"
    audio_path = os.path.join(user_folder, audio.filename)
    audio.save(audio_path)

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, fp16=False)
    text = result["text"]

    doc_path = os.path.join(user_folder, filename)
    doc = Document()
    doc.add_paragraph(text)
    doc.save(doc_path)

    history = load_history(history_file)
    history.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "original_name": audio.filename,
        "doc_name": filename,
        "model": model_name,
        "text": text
    })
    write_history(history, history_file)

    response = make_response(render_template("index.html", text=text, history=history))
    response.set_cookie("user_id", user_id)
    return response

@app.route("/view/<filename>")
def view_text(filename):
    user_id = get_user_id()
    _, history_file = get_user_paths(user_id)
    history = load_history(history_file)
    for item in history:
        if item["doc_name"] == filename:
            return render_template("view.html", item=item)
    return "見つかりませんでした", 404

@app.route("/download/<filename>")
def download(filename):
    user_id = get_user_id()
    user_folder, _ = get_user_paths(user_id)
    path = os.path.join(user_folder, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "ファイルが見つかりません", 404

@app.route("/delete-multiple", methods=["POST"])
def delete_multiple():
    user_id = get_user_id()
    user_folder, history_file = get_user_paths(user_id)
    selected = request.form.getlist("selected_files")
    history = load_history(history_file)
    history = [item for item in history if item["doc_name"] not in selected]
    for filename in selected:
        path = os.path.join(user_folder, filename)
        if os.path.exists(path):
            os.remove(path)
    write_history(history, history_file)
    return redirect(url_for("index"))

@app.route("/delete-all", methods=["POST"])
def delete_all():
    user_id = get_user_id()
    user_folder, history_file = get_user_paths(user_id)
    if os.path.exists(user_folder):
        for f in os.listdir(user_folder):
            os.remove(os.path.join(user_folder, f))
    write_history([], history_file)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
