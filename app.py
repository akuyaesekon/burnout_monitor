from flask import Flask, render_template, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset from the URL
url = 'https://raw.githubusercontent.com/sohansputhran/Will-your-employees-leave-you/refs/heads/master/Test.csv'
df = pd.read_csv(url)

# Print column names and first few rows for inspection
print(df.columns)
print(df.head())

# Ensure correct column names and data types
df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace

# Define a function to safely convert columns to numeric
def safe_convert_to_numeric(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    else:
        print(f"Column '{column_name}' does not exist in the dataset.")

# Update the column names based on the dataset inspection
safe_convert_to_numeric(df, 'Age')
safe_convert_to_numeric(df, 'Time_of_service')
safe_convert_to_numeric(df, 'Time_since_promotion')
safe_convert_to_numeric(df, 'growth_rate')
safe_convert_to_numeric(df, 'Travel_Rate')
safe_convert_to_numeric(df, 'Post_Level')
safe_convert_to_numeric(df, 'Pay_Scale')
safe_convert_to_numeric(df, 'Compensation_and_Benefits')
safe_convert_to_numeric(df, 'Work_Life_balance')
safe_convert_to_numeric(df, 'VAR1')
safe_convert_to_numeric(df, 'VAR2')
safe_convert_to_numeric(df, 'VAR3')
safe_convert_to_numeric(df, 'VAR4')
safe_convert_to_numeric(df, 'VAR5')
safe_convert_to_numeric(df, 'VAR6')
safe_convert_to_numeric(df, 'VAR7')

# Convert categorical columns to string
df['sales'] = df['sales'].astype(str) if 'sales' in df.columns else 'N/A'
df['salary'] = df['salary'].astype(str) if 'salary' in df.columns else 'N/A'

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define the target variable and features
target = 'Work_Life_balance'

# Handle NaN values in the target variable
df = df.dropna(subset=[target])

# Update features after dropping NaN values in the target variable
features = df.drop(columns=[target, 'Employee_ID'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)

# Function to generate recommendations
def generate_recommendations():
    recommendations = []
    for feature in feature_importances.index[:5]:  # Top 5 influential factors
        if feature == 'Time_of_service':
            recommendations.append("Consider providing more support to employees with longer service times.")
        elif feature == 'Pay_Scale':
            recommendations.append("Review and adjust pay scales to ensure fair compensation.")
        elif feature == 'Compensation_and_Benefits':
            recommendations.append("Enhance compensation and benefits packages.")
        elif feature == 'Post_Level':
            recommendations.append("Evaluate job roles and responsibilities to ensure they are manageable.")
        elif feature == 'Travel_Rate':
            recommendations.append("Reduce travel requirements for employees.")
        # Add more recommendations based on other influential factors
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    burnout_rate = pio.to_html(px.histogram(df, x='Work_Life_balance', title='Burnout Rate', labels={'Work_Life_balance': 'Work Life Balance'}, color_discrete_sequence=['#636EFA']), full_html=False)
    workload_correlation = pio.to_html(px.scatter(df, x='Time_of_service', y='Work_Life_balance', color='Work_Life_balance', title='Workload vs Burnout Level', labels={'Time_of_service': 'Time of Service', 'Work_Life_balance': 'Work Life Balance'}, color_continuous_scale=px.colors.sequential.Viridis), full_html=False)
    burnout_by_gender = pio.to_html(px.histogram(df, x='Gender', color='Work_Life_balance', title='Burnout by Gender', labels={'Gender': 'Gender', 'Work_Life_balance': 'Work Life Balance'}, color_discrete_sequence=px.colors.qualitative.Pastel), full_html=False)
    burnout_by_age = pio.to_html(px.histogram(df, x='Age', color='Work_Life_balance', title='Burnout by Age', labels={'Age': 'Age', 'Work_Life_balance': 'Work Life Balance'}, color_discrete_sequence=px.colors.qualitative.Set3), full_html=False)
    burnout_trends_by_seniority = pio.to_html(px.line(df, x='Time_of_service', y='Work_Life_balance', color='Work_Life_balance', title='Burnout Trends by Seniority', labels={'Time_of_service': 'Time of Service', 'Work_Life_balance': 'Work Life Balance'}, color_discrete_sequence=px.colors.qualitative.Bold), full_html=False)
    
    # Filter numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    heatmap = pio.to_html(px.imshow(numeric_df.corr(), text_auto=True, title='Heatmap of Correlations', color_continuous_scale=px.colors.sequential.Plasma), full_html=False)
    
    return render_template('dashboard.html', burnout_rate=burnout_rate, workload_correlation=workload_correlation, burnout_by_gender=burnout_by_gender, burnout_by_age=burnout_by_age, burnout_trends_by_seniority=burnout_trends_by_seniority, heatmap=heatmap)

@app.route('/recommendations')
def recommendations():
    recommendations = generate_recommendations()
    burnout_risk_factors = pio.to_html(px.bar(feature_importances.head(5), title='Top 5 Burnout Risk Factors', labels={'index': 'Factors', 'value': 'Importance'}, color_discrete_sequence=['#EF553B']), full_html=False)
    return render_template('recommendations.html', feature_importances=feature_importances, recommendations=recommendations, burnout_risk_factors=burnout_risk_factors)

@app.route('/additional-visualizations')
def additional_visualizations():
    burnout_by_work_mode = pio.to_html(px.histogram(df, x='salary', color='Work_Life_balance', title='Burnout by Work Mode', labels={'salary': 'Salary', 'Work_Life_balance': 'Work Life Balance'}, color_discrete_sequence=px.colors.qualitative.Safe), full_html=False)
    burnout_trends_by_seniority = pio.to_html(px.line(df, x='Time_of_service', y='Work_Life_balance', color='Work_Life_balance', title='Burnout Trends by Seniority', labels={'Time_of_service': 'Time of Service', 'Work_Life_balance': 'Work Life Balance'}, color_discrete_sequence=px.colors.qualitative.Bold), full_html=False)
    
    # Filter numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    heatmap = pio.to_html(px.imshow(numeric_df.corr(), text_auto=True, title='Heatmap of Correlations', color_continuous_scale=px.colors.sequential.Plasma), full_html=False)
    
    return render_template('additional_visualizations.html', burnout_by_work_mode=burnout_by_work_mode, burnout_trends_by_seniority=burnout_trends_by_seniority, heatmap=heatmap)

@app.route('/api/data')
def get_data():
    data = df.to_dict(orient='records')
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=2025)