from flask import Flask, render_template, request, send_file
import whisper
import os
from werkzeug.utils import secure_filename
from docx import Document
import uuid

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULT_FOLDER"] = "results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

model = whisper.load_model("small")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return "ファイルが見つかりません", 400

    file = request.files["audio"]
    if file.filename == "":
        return "ファイルが選択されていません", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = model.transcribe(filepath, fp16=False)
    text = result["text"]

    # Wordファイルを作成
    doc = Document()
    doc.add_paragraph(text)

    doc_id = str(uuid.uuid4())
    output_path = os.path.join(app.config["RESULT_FOLDER"], f"{doc_id}.docx")
    doc.save(output_path)

    return send_file(output_path, as_attachment=True, download_name="transcription.docx")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
