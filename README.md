# 🩸 Anemia Sense – Machine Learning Based Anemia Prediction System

Anemia Sense is a full-stack machine learning web application that predicts whether a person is affected by anemia based on key blood parameters. The system leverages Machine Learning for prediction and Flask for web deployment, providing an easy-to-use interface for users.

---

## 📌 Project Overview

Anemia is a common blood disorder caused by a deficiency of red blood cells or hemoglobin. Early detection plays a crucial role in effective treatment.  
This project uses a supervised machine learning model to analyze blood test values and predict anemia accurately.

---

## 🎯 Objectives

- To build a machine learning model for anemia prediction  
- To deploy the model using Flask  
- To provide a simple and interactive web interface  
- To demonstrate end-to-end ML project development  

---

## 🛠️ Technologies Used

- Programming Language: Python  
- Machine Learning: Scikit-learn  
- Web Framework: Flask  
- Frontend: HTML, CSS  
- Libraries: Pandas, NumPy  
- Version Control: Git & GitHub  
- Development Environment: GitHub Codespaces  

---

## 🧪 Input Parameters

- Hemoglobin  
- RBC (Red Blood Cell count)  
- PCV (Packed Cell Volume)  
- MCV (Mean Corpuscular Volume)  
- MCH (Mean Corpuscular Hemoglobin)  
- MCHC (Mean Corpuscular Hemoglobin Concentration)  

---

## ⚙️ How the Project Works

1. User enters blood test values through the web form  
2. Input data is sent to the Flask backend  
3. The trained machine learning model processes the input  
4. The system predicts:
   - Anemia Detected  
   - No Anemia  
5. Result is displayed on the web page  

---

## ▶️ How to Run the Project 

cd Flask
pip install flask pandas numpy scikit-learn
python app.py


Open Port 5000 in the browser to view the application.

---

## 📊 Model Details

- Algorithm Used: Logistic Regression  
- Model Type: Binary Classification  
- Output:
  - 1 → Anemia Detected  
  - 0 → No Anemia  

The trained model is saved as `model.pkl`.

---

## 🌐 Deployment

The application is executed using GitHub Codespaces, which allows cloud-based development and execution without local setup.

---

## ✅ Results

- Accurate prediction of anemia based on blood parameters  
- Clean and responsive user interface  
- Fully functional end-to-end ML pipeline  

---

## 🔮 Future Enhancements

- Use a larger real-world medical dataset  
- Improve model accuracy with advanced algorithms  
- Add user authentication  
- Deploy on cloud platforms like AWS or Render  

---

## 👩‍💻 Author
Poorvika A C  

