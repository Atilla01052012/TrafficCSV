import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Traffic Prediction", page_icon="🚦")

st.title("🚦 Traffic Level Prediction")
st.write("CSV faylını yükləyin və model nəticələrini görün.")

uploaded_file = st.file_uploader(
    "traffic.csv faylını seçin",
    type=["csv"]
)

if uploaded_file is not None:

    # CSV faylını oxuyuruq
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    # ID sütununu silirik
    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    # DateTime sütununu emal edirik
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.day
    df["weekday"] = df["DateTime"].dt.weekday

    df.drop("DateTime", axis=1, inplace=True)

    # Traffic level yaradırıq
    df["traffic_level"] = pd.cut(
        df["Vehicles"],
        bins=[0, 20, 40, 300],
        labels=[0, 1, 2]
    )

    # Feature və target
    x = df.drop("traffic_level", axis=1)
    y = df["traffic_level"]

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    # Accuracy
    accuracy = model.score(x_test, y_test)

    st.subheader("Model Accuracy")
    st.success(f"{accuracy:.2%}")

    # Prediction
    y_pred = model.predict(x_test)

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    st.dataframe(
        pd.DataFrame(
            cm,
            columns=["Pred 0", "Pred 1", "Pred 2"],
            index=["Actual 0", "Actual 1", "Actual 2"]
        )
    )

    st.subheader("Processed Dataset")
    st.dataframe(df.head())