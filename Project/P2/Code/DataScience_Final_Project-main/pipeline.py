import subprocess

subprocess.run(["python", "scripts/init_db.py"])
subprocess.run(["python", "scripts/preprocess.py"])
subprocess.run(["python", "scripts/feature_engineering.py"])

