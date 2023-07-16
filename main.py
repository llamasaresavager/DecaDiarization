import subprocess

def run_fastapi():
    subprocess.run(["uvicorn", "app:app", "--reload"])

def run_ui():
    subprocess.run(["streamlit", "run", "ui.py"])

if __name__ == "__main__":
    import threading

    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    run_ui()
