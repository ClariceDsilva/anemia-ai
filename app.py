from flask import Flask, render_template, request
import os
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = YOLO("best.pt")

# -------------------------------
# IMAGE PREDICTION FUNCTION
# -------------------------------
def predict_images(image_paths):
    anemia_score = 0
    total = 0

    for path in image_paths:
        results = model(path)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]

                total += 1

                if class_name == "pale nails":
                    anemia_score += conf

                elif class_name == "conjunctiva":
                    anemia_score += (1 - conf)

    if total == 0:
        return 0.5

    return anemia_score / total


# -------------------------------
# QUESTION SCORE
# -------------------------------
def calculate_question_score(form):
    questions = [
        "q1","q2","q3","q4","q5",
        "q6","q7","q8","q9","q10"
    ]

    score = 0
    for q in questions:
        if form.get(q) == "yes":
            score += 1

    return score / len(questions)


# -------------------------------
# FINAL DECISION
# -------------------------------
def final_decision(img_score, q_score):
    final_score = (img_score + q_score) / 2

    if final_score > 0.7:
        return {
            "status": "⚠️ High Risk of Anemia",
            "message": "You may have strong signs of anemia.",
            "advice": "Please consult a doctor immediately.",
            "confidence": int(final_score * 100)
        }

    elif final_score > 0.4:
        return {
            "status": "⚠️ Moderate Risk",
            "message": "You may have mild signs of anemia.",
            "advice": "Consider improving diet and consult doctor if needed.",
            "confidence": int(final_score * 100)
        }

    else:
        return {
            "status": "✅ Low Risk",
            "message": "No major signs of anemia detected.",
            "advice": "Maintain a healthy lifestyle and regular checkups.",
            "confidence": int(final_score * 100)
        }


# -------------------------------
# ROUTE
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        files = request.files.getlist("images")
        image_paths = []

        for file in files:
            if file.filename != "":
                path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(path)
                image_paths.append(path)

        img_score = predict_images(image_paths)
        q_score = calculate_question_score(request.form)

        result = final_decision(img_score, q_score)

        return render_template(
            "result.html",
            images=image_paths,
            result=result,
            img_score=round(img_score, 2),
            q_score=round(q_score, 2)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)