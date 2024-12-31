import webbrowser

# Path to your MLflow tracking URI
mlruns_path = "mlruns"

# Port and host settings
host = "0.0.0.0"
port = 5050


# Open the MLflow UI in the default browser
def open_mlflow_ui():
    url = f"http://{host}:{port}"
    webbrowser.open(url)


# Open in browser
open_mlflow_ui()
