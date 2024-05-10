from fastapi import FastAPI, File, UploadFile, Query
import subprocess
import os

app = FastAPI()

@app.post("/predict-height/")
async def predict_height(file: UploadFile = File(...)):
    # Save the uploaded file
    file_path = os.path.join("temp_images", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Call heightdetection.py script
    command = f"python3 heightdetection.py -i {file_path} -r 128"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    # Remove the uploaded image file
    os.remove(file_path)

    # Check for errors
    if error:
        return {"error": error.decode()}
    
    # Extract predicted height from output
    predicted_height = float(output.decode().split(":")[-1].strip())
    
    # Return predicted height
    return {"predicted_height_cm": predicted_height}
