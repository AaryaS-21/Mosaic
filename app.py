from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from quiz import*
import random


app = Flask(__name__)

df = pd.read_csv("dataset.csv")
#data.head()
#X = data.drop(["diagnosis", "id"], axis=1)
#Y = data["diagnosis"]


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from quiz import Quiz

# Drop non-numerical columns
df_numerical = df.select_dtypes(include=['number']).dropna()

# Separate the features and target variable
X = df_numerical.drop(['GRADE', 'Total salary if available', 'Accommodation type in Cyprus', 'COURSE ID'], axis=1)
y = df_numerical['GRADE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define the models
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Machine": SVR()
}

# Initialize dictionary to store the RMSE for each model
rmse_scores = {}

# Train each model and evaluate the RMSE on the test set
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse_scores[name] = sqrt(mean_squared_error(y_test, y_pred))

print(rmse_scores)



test_set_predictions = models["Random Forest"].predict(X_test_scaled)
print(test_set_predictions[:5])


def predict_grade(input_factors):
    # Convert input factors to DataFrame
    input_df = pd.DataFrame([input_factors])

    # Ensure the columns match the training features order
    input_df = input_df[X.columns]

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Predict the grade using the Random Forest model
    predicted_grade = models["Random Forest"].predict(input_scaled)

    return predicted_grade[0]

# Example usage of the function with an example input


# Call the function with the example input_factors




@app.route('/')
def home():  # put application's code here
    return render_template("index.html")


@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")


@app.route('/prediction', methods=["GET", "POST"])
def prediction():

    predicted_grade = None

    if request.method == "POST":
        age = request.form.get('age')
        gender = request.form.get('gender')
        school = request.form.get('school')
        scholarship = request.form.get('scholarship')
        additionalwork = request.form.get('additionalwork')
        artsport = request.form.get('artsport')
        partner = request.form.get('partner')
        seminars = request.form.get('seminars')
        transport = request.form.get('transport')
        medu = request.form.get('medu')
        fedu = request.form.get('fedu')
        siblings = request.form.get('siblings')
        parentalt = request.form.get('parentalt')
        moc = request.form.get('moc')
        doc = request.form.get('doc')
        studyhr = request.form.get('studyhr')
        readnosci = request.form.get('readnosci')
        readsci = request.form.get('readsci')
        impact = request.form.get('impact')
        attendance = request.form.get('attendance')
        prepone = request.form.get('prepone')
        preptwo = request.form.get('preptwo')
        notes = request.form.get('notes')
        listen = request.form.get('listen')
        discuss = request.form.get('discuss')
        flip = request.form.get('request')
        gpalast = request.form.get('gpalast')
        gpaexp = request.form.get('gpaexp')

        input_factors = {
            'Student Age' : age,
            'Sex' : gender,
            'Graduated high-school type' : school,
            'Scholarship type' : scholarship,
            'Additional work': additionalwork,
            'Regular artistic or sports activity' : artsport,
            'Do you have a partner' : partner,
            'Attendance to the seminars/conferences related to the department' : seminars,
            'Transportation to the university' : transport,
            'Mother’s education' : medu,
            'Father’s education ' : fedu,
            'Number of sisters/brothers' : siblings,
            'Parental status' : parentalt,
            'Mother’s occupation' : moc,
            'Father’s occupation' : doc,
            'Weekly study hours' : studyhr,
            'Reading frequency' : readnosci,
            'Reading frequency.1' : readsci,
            'Impact of your projects/activities on your success' : impact,
            'Attendance to classes' : attendance,
            'Preparation to midterm exams 1' : prepone,
            'Preparation to midterm exams 2' : preptwo,
            'Taking notes in classes' : notes,
            'Listening in classes' : listen,
            'Discussion improves my interest and success in the course' : discuss,
            'Flip-classroom' : flip,
            'Cumulative grade point average in the last semester (/4.00)' : gpalast,
            'Expected Cumulative grade point average in the graduation (/4.00)' : gpaexp
        }

        predicted_grade = predict_grade(input_factors)
        print(f"Predicted Grade: {predicted_grade}")



        
    
    return render_template("prediction.html", predicted_grade=predicted_grade)


@app.route('/personality',methods=["GET", "POST"])
def personality():
    
    if request.method == "POST": 
        qn1 = int(request.form.get('qn1'))
        qn2 = int(request.form.get('qn2'))
        qn3 = int(request.form.get('qn3'))
        qn4 = int(request.form.get('qn4'))
        qn5 = int(request.form.get('qn5'))
        qn6 = int(request.form.get('qn6'))
        qn7 = int(request.form.get('qn7'))
        qn8 = int(request.form.get('qn8'))
        qn9 = int(request.form.get('qn9'))
        qn10 = int(request.form.get('qn10'))
        qn11 = int(request.form.get('qn11'))
        qn12 = int(request.form.get('qn12'))
        qn13 = int(request.form.get('qn13'))
        qn14 = int(request.form.get('qn14'))
        qn15 = int(request.form.get('qn15'))
        qn16 = int(request.form.get('qn16'))
        qn17 = int(request.form.get('qn17'))
        qn18 = int(request.form.get('qn18'))
        qn19 = int(request.form.get('qn19'))
        qn20 = int(request.form.get('qn20'))
        qn21 = int(request.form.get('qn21'))
        qn22 = int(request.form.get('qn22'))
        qn23 = int(request.form.get('qn23'))
        qn24 = int(request.form.get('qn24'))
        qn25 = int(request.form.get('qn25'))
        qn26 = int(request.form.get('qn26'))
        qn27 = int(request.form.get('qn27'))
        qn28 = int(request.form.get('qn28'))
        qn29 = int(request.form.get('qn29'))
        qn30 = int(request.form.get('qn30'))
        qn31 = int(request.form.get('qn31'))
        qn32 = int(request.form.get('qn32'))
        qn33 = int(request.form.get('qn33'))
        qn34 = int(request.form.get('qn34'))
        qn35 = int(request.form.get('qn35'))
        qn36 = int(request.form.get('qn36'))
        qn37 = int(request.form.get('qn37'))
        qn38 = int(request.form.get('qn38'))
        qn39 = int(request.form.get('qn39'))
        qn40 = int(request.form.get('qn40'))
        questions = [qn1, qn2, qn3, qn4, qn5, qn6, qn7, qn8, qn9, qn10, qn11, qn12, qn13, qn14, qn15, qn16, qn17, qn18, qn19, qn20, qn21, qn22, qn23, qn24, qn25, qn26, qn27, qn28, qn29, qn30, qn31, qn32, qn33, qn34, qn35, qn36, qn37, qn38, qn39, qn40]
        q = Quiz(questions)
        print(q)
        return redirect(url_for("output"))
    return render_template("personality.html")

@app.route("/output")
def output():
    advocates = ["Lawyer", "Human Resources Manager", "Social Worker", "Nonprofit Administrator", "Environmental Activist", "Immigration Lawyer", "Labor Relations Specialist"]
    humanitarians = ["Teacher","Counselor","Doctor","Social Worker","Community Organizer","Mental Health Counselor","Public Health Educator","International Aid Worker"]
    dreamer = ["Artist", "Muscian", "Writer", "Chef", "Fashion Designer", "Filmmaker", "Interior Designer", "Creative Director"]
    critical = ["Mathematician", "Economist", "Market Research Analyst", "Financial Advisor", "Operations Research Analyst", "Statistician"]
    econ = ["Sales Manager", "Business Owner", "Real Estate Agent", "Investment Banker", "Startup Founder"]
    problemsolver = ["Scientist","Engineer","Software Developer","Data Analyst","Financial Analyst","Quality Assurance","Engineer","Systems Analyst","Forensic Scientist"]

    advocate = random.choice(advocates)
    humanitarian = random.choice(humanitarians)
    dream = random.choice(dreamer)
    critic = random.choice(critical)
    economic = random.choice(econ)
    problem_solving = random.choice(problemsolver)


    return render_template("output.html", advocate=advocate, humanitarian=humanitarian, dream=dream, critic=critic, economic=economic, problem_solving=problem_solving)




if __name__ == '__main__':
    app.run(port=5000, debug=True)