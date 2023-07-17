import subprocess
import threading
import time

FASTAPI_CMD = ["uvicorn", "app:app", "--reload"]
UI_CMD = ["streamlit", "run", "ui.py"]

def run_process(cmd):
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute command: {e.cmd}. Exit status: {e.returncode}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_process, args=(FASTAPI_CMD,))
    ui_thread = threading.Thread(target=run_process, args=(UI_CMD,))

    fastapi_thread.start()
    ui_thread.start()

    try:
        while True:  # keep the main thread alive to allow for manual interrupt
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
